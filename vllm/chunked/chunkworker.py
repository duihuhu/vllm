from typing import List, Tuple, Dict
import torch
from transformers import PreTrainedTokenizerBase
import time

from vllm.config import ModelConfig, ParallelConfig
from vllm.chunked.chunk import Chunk, Sequence, ChunkStatus, ChunkSamplingParams, ChunkInputMetadata
from vllm.model_executor import get_model, set_random_seed
from vllm.utils import random_uuid, Counter
from vllm.chunked.chunkcache import ChunkCacheBlocks
#from vllm.engine.ray_utils import initialize_cluster
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel, initialize_all_reduce_launcher)

class ChunkWorker:
    def __init__(self,
                 chunk_size: int,
                 chunk_num: int,
                 model_config: ModelConfig,
                 predict_model_config: ModelConfig,
                 parallel_config: ParallelConfig,
                 rank: int,
                 distributed_init_method: str,
                 ) -> None:
        self.chunk_size = chunk_size
        self.chunk_num = chunk_num
        self.model_config = model_config
        self.predict_model_config = predict_model_config
        self.parallel_config = parallel_config
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.total_sequences: List[Sequence] = []
        self.job_sequences: Dict[str, Sequence] = {}
        self.job_chunks: List[Chunk] = []
        self.counter = Counter()
        self._set_self_model()
        self._set_self_kv_cache()
        #self._set_self_cacheblock()
    
    def _set_self_model(self) -> None:
        #distributed_init_method, _ = initialize_cluster(self.parallel_config)
        self._init_distributed_environment(parallel_config = self.parallel_config, 
                                           rank = self.rank, 
                                           distributed_init_method = self.distributed_init_method)
        set_random_seed(seed = self.model_config.seed)
        self.model = get_model(model_config = self.model_config,
                               Chunked = True)
        self.predict_model = get_model(model_config = self.predict_model_config,
                               Predicted = True)
        self.num_layers = self.predict_model_config.get_num_layers(self.parallel_config)
        initialize_all_reduce_launcher(
            4096,
            #self.chunk_size,
            #2560,
            self.model_config.get_hidden_size(),
            self.model_config.dtype,
        )

    def _init_distributed_environment(self, parallel_config: ParallelConfig, rank: int,
        distributed_init_method: str) -> None:
        """Initialize the distributed environment."""
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )
        # A small all_reduce for warmup.
        torch.distributed.all_reduce(torch.zeros(1).cuda())
        initialize_model_parallel(parallel_config.tensor_parallel_size,
                                parallel_config.pipeline_parallel_size)

    def _set_self_kv_cache(self) -> None:
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        num_tokens = self.chunk_size
        head_size= self.model_config.get_head_size()
        num_heads = self.model_config.get_num_heads(self.parallel_config)
        hidden_states = num_heads * head_size
        dtype = self.model_config.dtype
        for _ in range(num_layers):
            k_block = torch.empty(
                size = (self.chunk_num, num_tokens, hidden_states),
                dtype = dtype,
                device = "cuda"
            )
            v_block = torch.empty(
                size = (self.chunk_num, num_tokens, hidden_states),
                dtype = dtype,
                device = "cuda"
            )
            kv_cache.append((k_block, v_block))
        self.kv_cache = kv_cache
    
    def _set_self_cacheblock(self) -> None:
        self.cacheblock = ChunkCacheBlocks(blocks_num = self.chunk_num)

    def add_requests(self, prompt_token_ids: List[int], 
                     sampling_params: ChunkSamplingParams) -> None:
        seq_id = random_uuid()
        now_time = time.time()
        self.total_sequences.append(Sequence(seq_id = seq_id, 
                                             prompt_token_ids = prompt_token_ids,
                                             sampling_params = sampling_params,
                                             start_time = now_time))
    
    # add error handle when extremly long req appears
    def set_job_sequences(self) -> None:
        total_token_num = self.chunk_num * self.chunk_size
        count = 0
        while len(self.total_sequences) > 0:
            sequence = self.total_sequences[0]
            if count + sequence.prompt_len <= total_token_num:
                sequence = self.total_sequences.pop(0)
                count += sequence.prompt_len
                self.job_sequences[sequence.seq_id] = sequence
            else:
                break
    
    def set_job_chunks(self) -> None:
        all_token_ids: List[int] = []
        all_token_seqs: List[str] = []
        for _, sequence in self.job_sequences.items():
            for token_id in sequence.prompt_token_ids:
                all_token_ids.append(token_id)
                all_token_seqs.append(sequence.seq_id)
        st = 0
        token_num = len(all_token_ids)
        while st < token_num:
            ed = st + self.chunk_size
            if ed >= token_num:
                ed = token_num
            chunk_id = self.counter.__next__()
            chunk = Chunk(chunk_id = chunk_id, chunk_size = self.chunk_size, 
                          chunk_status = ChunkStatus.WAITING)
            temp_seqtoken_count: Dict[str, int] = {}
            for i in range(st, ed):
                chunk.prompt_token_ids.append(all_token_ids[i])
                temp_seqtoken_count[all_token_seqs[i]] = temp_seqtoken_count.setdefault(all_token_seqs[i], 0) + 1
                self.job_sequences[all_token_seqs[i]].chunks_to_prompts[chunk_id] = self.job_sequences[all_token_seqs[i]].chunks_to_prompts.setdefault(chunk_id, 0) + 1
            for temp_seq_id, temp_token_len in temp_seqtoken_count.items():
                chunk.raw_sequence_ids.append(temp_seq_id)
                chunk.prompt_lens.append(temp_token_len)
                ans = 0
                for temp_chunk_id, used_token_num in self.job_sequences[temp_seq_id].chunks_to_prompts.items():
                    if temp_chunk_id == chunk_id:
                        break
                    else:
                        ans += used_token_num
                chunk.kv_prefixs.append(ans)
            chunk.set_seqs_to_lens_and_prefixs()
            temp_block = self.cacheblock.allocate_block()
            chunk.set_self_block(block = temp_block)
            self.job_chunks.append(chunk)
            st += self.chunk_size
    
    def reduce_outputs(self) -> None:
        #for _, sequence in self.job_sequences.items():
        #    for i in range(len(sequence.outputs)):
        #        if i != 0:
        #            sequence.outputs[0] = torch.cat((sequence.outputs[0], sequence.outputs[i]), 0)
        # free all chunks' cache
        for chunk in self.job_chunks:
            self.cacheblock.free_block(block = chunk.cache_block)
            chunk.chunk_status = ChunkStatus.PREFILLED
    
    def generate_first_token_id(self) -> None:
        for _, sequence in self.job_sequences.items():
            st = time.time()
            output_tokens_list, logprob = self._execute_sampler(logits = sequence.outputs[0], 
                                                                sampling_params = sequence.sampling_params)
            #logprob_index = self._execute_sampler(logits = sequence.outputs[0], 
            #                                                    sampling_params = sequence.sampling_params)
            ed = time.time()
            sequence.add_sampler_time(st, ed)
            sequence.add_first_token_id(output_tokens_list[0])
            sequence.add_first_token_logprob(logprob)

    def generate_first_token_str(self, tokenizer: PreTrainedTokenizerBase) -> None:
        for _, sequence in self.job_sequences.items():
            input_new_token = []
            new_token = tokenizer.convert_ids_to_tokens(sequence.first_token_id, skip_special_tokens = True)
            input_new_token.extend(new_token)
            new_output_text = tokenizer.convert_tokens_to_string(input_new_token)
            sequence.add_first_token_str(new_output_text = new_output_text)
    
    @torch.inference_mode()
    def execute_sampler(self, 
                         logits: torch.Tensor, 
                         sampling_params: List[ChunkSamplingParams]) -> Tuple[List[int], List[float]]:
        output_tokens_list, logprobs = self.model.sampler(self.model.lm_head_weight, logits, sampling_params)
        #logprob_index = self.model.sampler(self.model.lm_head_weight, logits, sampling_params)
        return (output_tokens_list, logprobs)

    def greedy_search(self) -> List[int]:
        ans: List[int] = []
        for _, sequence in self.job_sequences.items():
            logits = sequence.outputs[0]
            logits = logits.reshape(1, -1)
            logits = torch.softmax(logits, dim = 1)
            max_prob_index = torch.argmax(logits, dim = 1)
            ans.append(max_prob_index)
        return ans
    
    @torch.inference_mode()
    def execute_model(self, 
                       inputs: torch.Tensor, 
                       inputs_positions: torch.Tensor, 
                       #kv_cache: List[Tuple[torch.Tensor, torch.Tensor]], 
                       chunkmetadata: ChunkInputMetadata) -> Tuple[List[int], List[float]]: #Tuple[torch.Tensor, float, float]:
        #start_time = time.time()
        #print(inputs.shape)
        #print(inputs_positions.shape)
        output = self.model(
            input_ids = inputs,
            positions = inputs_positions,
            kv_caches = self.kv_cache,
            cache_events = None,
            chunkinputmetadata = chunkmetadata
        )
        #end_time = time.time()
        return output #(output, start_time, end_time)
    
    @torch.inference_mode()
    def execute_predict_model(self,
                              inputs: torch.Tensor,
                              inputs_positions: torch.Tensor,
                              chunkmetadata: ChunkInputMetadata) -> List[int]:
        kv_cache = [(None, None)] * self.num_layers
        output = self.predict_model(
            input_ids = inputs,
            positions = inputs_positions,
            kv_caches = kv_cache,
            cache_events = None,
            chunkinputmetadata = chunkmetadata
        )
        return output