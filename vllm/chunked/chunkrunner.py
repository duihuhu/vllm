from transformers import PreTrainedTokenizerBase
import json
from typing import List, Dict, Tuple, Any
import torch
import time
import random

from vllm.config import ModelConfig, ParallelConfig
from vllm.chunked.chunkworker import ChunkWorker
from vllm.chunked.chunkcache import ChunkCacheBlocks
from vllm.chunked.chunk import Chunk, ChunkInputMetadata, ChunkSamplingParams, ChunkStatus, Sequence
from vllm.worker.worker import _pad_to_max
from vllm.engine.ray_utils import initialize_cluster, ray
from vllm.utils import random_uuid, Counter

class ChunkRunner:
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 chunk_size: int,
                 chunk_num: int) -> None:
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_num = chunk_num
        self.all_total_sequences: List[Sequence] = []
        self.all_job_sequences: Dict[str, Sequence] = {}
        self.all_job_chunks: List[Chunk] = []
        self.counter = Counter()
        self.cacheblock = ChunkCacheBlocks(blocks_num = self.chunk_num)

    def _add_requests(self, 
                      prompt_token_ids: List[int], 
                      sampling_params: ChunkSamplingParams) -> None:
        seq_id = random_uuid()
        now_time = time.time()
        self.all_total_sequences.append(Sequence(seq_id = seq_id, 
                                             prompt_token_ids = prompt_token_ids,
                                             sampling_params = sampling_params,
                                             start_time = now_time)) 

    def set_self_configs(self, model: str, tensor_parallel_size: int) -> None:
        model_config = ModelConfig(model = model, tokenizer = None, tokenizer_mode = 'auto', 
                                   trust_remote_code = False, download_dir = None, use_np_weights = False,
                                   use_dummy_weights = False, dtype = 'auto', seed = 0)
        self.model_config = model_config
        if tensor_parallel_size > 1:
            worker_use_ray = True
        else:
            worker_use_ray = False
        self.worker_use_ray = worker_use_ray
        self.parallel_config = ParallelConfig(pipeline_parallel_size = 1,
                                              tensor_parallel_size = tensor_parallel_size,
                                              worker_use_ray = self.worker_use_ray)
        self._set_self_ray_environment()
    
    def _set_self_ray_environment(self) -> None:
        distributed_init_method, devices = initialize_cluster(parallel_config = self.parallel_config,
                                                              engine_use_ray = self.worker_use_ray)
        self.distributed_init_method = distributed_init_method
        self.devices = devices
    
    def set_self_chunkworker(self, chunk_size: int, chunk_num: int) -> None:
        chunk_worker = ChunkWorker(chunk_size = chunk_size, chunk_num = chunk_num, 
                                   model_config = self.model_config,
                                   parallel_config = self.parallel_config,
                                   rank = 0,
                                   distributed_init_method = self.distributed_init_method)
        self.chunk_worker = chunk_worker
    
    def set_parallel_chunkworkers(self) -> None:
        self.workers: List[ChunkWorker] = []
        assert len(self.devices) == 1, "PP is under coding"
        for rank, node_resource, _ in self.devices[0]:
            worker_cls = ChunkWorker
            if self.worker_use_ray:
                worker_cls = ray.remote(num_cpus = 0,
                                        num_gpus = 1,
                                        resources = {node_resource: 1e-3})(worker_cls).remote
            worker = worker_cls(chunk_size = self.chunk_size,
                                chunk_num = self.chunk_num,
                                model_config = self.model_config,
                                parallel_config = self.parallel_config,
                                rank = rank,
                                distributed_init_method = self.distributed_init_method)
            self.workers.append(worker)
                
    
    def set_inputs(self, dataset_path: str, num_requests: int) -> None:
        with open(dataset_path) as f:
            dataset = json.load(f)
        dataset = [
            data for data in dataset
            if len(data["conversations"]) >= 2
        ]
        dataset = [
            (data["conversations"][0]["value"], data["conversations"][1]["value"])
            for data in dataset
        ]
        prompts = [prompt for prompt, _ in dataset]
        prompt_token_ids = self.tokenizer(prompts).input_ids
        completions = [completion for _, completion in dataset]
        completion_token_ids = self.tokenizer(completions).input_ids
        tokenized_dataset = []
        for i in range(len(dataset)):
            output_len = len(completion_token_ids[i])
            tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))
        filtered_dataset = []
        num_requests_count = 0
        for _, prompt_token_ids, output_len in tokenized_dataset:
            prompt_len = len(prompt_token_ids)
            if prompt_len < 4 or output_len < 4:
                continue
            filtered_dataset.append(prompt_token_ids)
            num_requests_count += 1
            if num_requests_count == num_requests:
                break
        self.requests = filtered_dataset
    
    def _add_requests_to_self(self) -> None:
        #for prompt_token_ids in self.requests:
        cold_start_token_ids = [random.randint(0, 100) for _ in range(self.chunk_size)]
        cold_start_sampling_params = ChunkSamplingParams(temperature = 0, top_p = 1.0, top_k = -1)
        self._add_requests(prompt_token_ids = cold_start_token_ids, 
                                       sampling_params = cold_start_sampling_params)
        prompt_lens = [41, 45, 40, 45, 45, 40, 256,
                       41, 43, 40, 46, 49, 44, 249, 
                       43, 40, 45, 48, 43, 49, 244, 
                       42, 44, 40, 42, 344]
        for prompt_len in prompt_lens:
            sampling_params = ChunkSamplingParams(temperature = 0, top_p = 1.0, top_k = -1)
            #self.chunk_worker.add_requests(prompt_token_ids = prompt_token_ids, sampling_params = sampling_params)
            dummy_prompt_token_ids = [random.randint(0, 100) for _ in range(prompt_len)]
            self._add_requests(prompt_token_ids = dummy_prompt_token_ids, sampling_params = sampling_params)
    
    def _start_worker(self) -> None:
        self._add_requests_to_self()
        self._set_job_sequences()
        self._set_job_chunks()
        #self.chunk_worker.set_job_sequences()
        #self.chunk_worker.set_job_chunks()

    def _prepare_model_inputs(self, 
                              chunk: Chunk) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, List[Tuple[int, int, int]]]]:
        chunk_size = self.chunk_size
        input_tokens = chunk.prompt_token_ids
        input_positions: List[int] = []
        for i, slice_length in enumerate(chunk.prompt_lens):
            slice_positions: List[int] = list(range(slice_length))
            slice_positions = [x + chunk.kv_prefixs[i] for x in slice_positions]
            input_positions.extend(slice_positions)
        input_tokens = _pad_to_max(input_tokens, max_len = chunk_size)
        input_tokens_tensor = torch.cuda.LongTensor(input_tokens)
        input_positions = _pad_to_max(input_positions, max_len = chunk_size)
        input_positions_tensor = torch.cuda.LongTensor(input_positions)
        kv_cache_ids: Dict[int, List[Tuple[int, int, int]]] = {} # prompt_len to (block, st, length)s
        for seq_id, kv_cache_num in chunk.seqs_to_prefixs.items():
            sequence = self.all_job_sequences[seq_id]
            if kv_cache_num == 0:
                prompt_len = chunk.seqs_to_lens.get(seq_id)
                kv_cache_ids.setdefault(prompt_len, [])
            else:
                for chunk_id, used_token_num in sequence.chunks_to_prompts.items():
                    if chunk_id == chunk.chunk_id:
                        break
                    else:
                        block_id = self.all_job_chunks[chunk_id].cache_block_id
                        st = 0
                        for a_seq_id, slice_token_num in self.all_job_chunks[chunk_id].seqs_to_lens.items():
                            if a_seq_id == seq_id:
                                break
                            else:
                                st += slice_token_num
                        prompt_len = chunk.seqs_to_lens.get(seq_id)
                        kv_cache_ids.setdefault(prompt_len, []).append((block_id, st, used_token_num))
        return (input_tokens_tensor, input_positions_tensor, kv_cache_ids)
    
    def run_worker(self) -> None:
        self._start_worker()

        #now_time = time.time()
        #print(f"Added in working pool at {now_time}")
        
        for chunk in self.all_job_chunks: #for chunk in self.chunk_worker.job_chunks:
            chunk.chunk_status = ChunkStatus.RUNNING
            input_tokens_tensor, input_positions_tensor, kv_cache_ids = self._prepare_model_inputs(chunk)
            chunkinputmetadata = ChunkInputMetadata(prompt_lens = chunk.prompt_lens, 
                                                    kv_prefixs = chunk.kv_prefixs,
                                                    kv_prefixs_blocks = kv_cache_ids, 
                                                    kv_block = chunk.cache_block_id)
            output = self._run_workers_model("execute_model",
                                        inputs = input_tokens_tensor,
                                        inputs_positions = input_positions_tensor,
                                        #kv_cache = self.chunk_worker.kv_cache,
                                        chunkmetadata = chunkinputmetadata)
            st = 0
            idxs: List[int] = []
            sampling_params: List[ChunkSamplingParams] = []
            do_sampling: List[str] = []
            for seq_id, prompt_len in chunk.seqs_to_lens.items():
                ed = st + prompt_len
                self.all_job_sequences[seq_id].update_count(prompt_len)
                #self.chunk_worker.job_sequences[seq_id].update_count(prompt_len)
                if self.all_job_sequences[seq_id].is_full(): #if self.chunk_worker.job_sequences[seq_id].is_full():
                    idxs.append(ed - 1)
                    do_sampling.append(seq_id)
                    sampling_params.append(self.all_job_sequences[seq_id].sampling_params)
                    #sampling_params.append(self.chunk_worker.job_sequences[seq_id].sampling_params)
                #self.chunk_worker.job_sequences[seq_id].outputs.append(output[st: ed])
                st = ed
                #self.chunk_worker.job_sequences[seq_id].add_start_and_end_time(st = start_time, ed = end_time)
                #st = ed
            output = output[idxs]
            output_token_list, logprobs = self._run_workers_sampler("execute_sampler",
                                                            logits = output, 
                                                            sampling_params = sampling_params)
            end_time = time.time()
            for i, id in enumerate(do_sampling):
                self.all_job_sequences[id].add_first_token_id(output_token_list[i])
                self.all_job_sequences[id].add_first_token_logprob(logprobs[i])
                self.all_job_sequences[id].set_end_time(end_time)
                '''self.chunk_worker.job_sequences[id].add_first_token_id(output_token_list[i])
                self.chunk_worker.job_sequences[id].add_first_token_logprob(logprobs[i])
                self.chunk_worker.job_sequences[id].set_end_time(end_time)'''

        self._reduce_outputs()    
        #self.chunk_worker.reduce_outputs()
        #self.chunk_worker.generate_first_token_id()
        #self.chunk_worker.generate_first_token_str(tokenizer = self.tokenizer)
    
    def _run_workers_model(
        self,
        method: str,
        *args,
        **kwargs
    ) -> Any:
        all_outputs = []
        for worker in self.workers:
            executor = getattr(worker, method)
            if self.parallel_config.worker_use_ray:
                executor = executor.remote
            
            output = executor(*args, **kwargs)
            all_outputs.append(output)
        
        if self.parallel_config.worker_use_ray:
            all_outputs = ray.get(all_outputs)
        
        concatenated_result = torch.cat(all_outputs, dim=-1)
        item0 = all_outputs[0]
        for item in all_outputs[1:]:
            print(item.shape)
            print(item0.eq(item))
        return concatenated_result
    
    def _run_workers_sampler(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        for worker in self.workers:
            executor = getattr(worker, method)
            if self.parallel_config.worker_use_ray:
                executor = executor.remote

            output = executor(*args, **kwargs)
            all_outputs.append(output)

        if self.parallel_config.worker_use_ray:
            all_outputs = ray.get(all_outputs)

        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output
    
    def _set_job_sequences(self) -> None:
        total_token_num = self.chunk_num * self.chunk_size
        count = 0
        while len(self.all_total_sequences) > 0:
            sequence = self.all_total_sequences[0]
            if count + sequence.prompt_len <= total_token_num:
                sequence = self.all_total_sequences.pop(0)
                count += sequence.prompt_len
                self.all_job_sequences[sequence.seq_id] = sequence
            else:
                break
    
    def _set_job_chunks(self) -> None:
        all_token_ids: List[int] = []
        all_token_seqs: List[str] = []
        for _, sequence in self.all_job_sequences.items():
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
                self.all_job_sequences[all_token_seqs[i]].chunks_to_prompts[chunk_id] = self.all_job_sequences[all_token_seqs[i]].chunks_to_prompts.setdefault(chunk_id, 0) + 1
            for temp_seq_id, temp_token_len in temp_seqtoken_count.items():
                chunk.raw_sequence_ids.append(temp_seq_id)
                chunk.prompt_lens.append(temp_token_len)
                ans = 0
                for temp_chunk_id, used_token_num in self.all_job_sequences[temp_seq_id].chunks_to_prompts.items():
                    if temp_chunk_id == chunk_id:
                        break
                    else:
                        ans += used_token_num
                chunk.kv_prefixs.append(ans)
            chunk.set_seqs_to_lens_and_prefixs()
            temp_block = self.cacheblock.allocate_block()
            chunk.set_self_block(block = temp_block)
            self.all_job_chunks.append(chunk)
            st += self.chunk_size
    
    def _reduce_outputs(self) -> None:
        #for _, sequence in self.job_sequences.items():
        #    for i in range(len(sequence.outputs)):
        #        if i != 0:
        #            sequence.outputs[0] = torch.cat((sequence.outputs[0], sequence.outputs[i]), 0)
        # free all chunks' cache
        for chunk in self.all_job_chunks:
            self.cacheblock.free_block(block = chunk.cache_block)
            chunk.chunk_status = ChunkStatus.PREFILLED