from typing import List, Tuple, Dict
import torch
from transformers import PreTrainedTokenizerBase

from vllm.config import ModelConfig, ParallelConfig
from vllm.chunked.chunk import Chunk, Sequence, ChunkStatus, ChunkSamplingParams
from vllm.model_executor import get_model, set_random_seed
from vllm.utils import random_uuid, Counter
from vllm.chunked.chunkcache import ChunkCacheBlocks
from vllm.transformers_utils.tokenizer import detokenize_incrementally
from vllm.engine.ray_utils import initialize_cluster
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel, initialize_all_reduce_launcher)

class ChunkWorker:
    def __init__(self,
                 chunk_size: int,
                 chunk_num: int,
                 model_config: ModelConfig) -> None:
        self.chunk_size = chunk_size
        self.chunk_num = chunk_num
        self.model_config = model_config
        self.parallel_config = ParallelConfig(pipeline_parallel_size = 1,
                                              tensor_parallel_size = 1,
                                              worker_use_ray = False)
        self.total_sequences: List[Sequence] = []
        self.job_sequences: Dict[str, Sequence] = {}
        self.job_chunks: List[Chunk] = []
        self.counter = Counter()
        self._set_self_model()
        self._set_self_kv_cache()
        self._set_self_cacheblock()
    
    def _set_self_model(self) -> None:
        distributed_init_method, _ = initialize_cluster(self.parallel_config)
        self._init_distributed_environment(parallel_config = self.parallel_config, 
                                           rank = 0, 
                                           distributed_init_method = distributed_init_method)
        set_random_seed(seed = self.model_config.seed)
        self.model = get_model(model_config = self.model_config)
        initialize_all_reduce_launcher(
            2560,
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
        self.total_sequences.append(Sequence(seq_id = seq_id, prompt_token_ids = prompt_token_ids,
                                             sampling_params = sampling_params))
    
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
        for _, sequence in self.job_sequences.items():
            for i in range(len(sequence.outputs)):
                if i != 0:
                    sequence.outputs[0] = torch.cat((sequence.outputs[0], sequence.outputs[i]), 0)
        # free all chunks' cache
        for chunk in self.job_chunks:
            self.cacheblock.free_block(block = chunk.cache_block)
            chunk.chunk_status = ChunkStatus.PREFILLED
    
    def generate_first_token_id(self) -> None:
        for _, sequence in self.job_sequences.items():
            output_tokens_list, logprob = self._execute_sampler(logits = sequence.outputs[0], 
                                                                sampling_params = sequence.sampling_params)
            sequence.add_first_token_id(output_tokens_list[0])
            sequence.add_first_token_logprob(logprob)
    
    def generate_first_token_str(self, tokenizer: PreTrainedTokenizerBase) -> None:
        for _, sequence in self.job_sequences.items():
            old_output_tokens: List[str] = []
            _, new_output_text = detokenize_incrementally(
                        tokenizer,
                        old_output_tokens,
                        sequence.first_token_id,
                        skip_special_tokens=True,
                    )
            sequence.add_first_token_str(new_output_text = new_output_text)
    
    @torch.inference_mode()
    def _execute_sampler(self, logits: torch.Tensor, sampling_params: ChunkSamplingParams) -> Tuple[List[int], float]:
        output_tokens_list, logprob = self.model.sampler(self.model.lm_head_weight, logits, sampling_params)
        return (output_tokens_list, logprob)