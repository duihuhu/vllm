from transformers import PreTrainedTokenizerBase
import json
from typing import List, Dict, Tuple
import torch
import time
import random

from vllm.config import ModelConfig
from vllm.chunked.chunkworker import ChunkWorker
from vllm.chunked.chunk import Chunk, ChunkInputMetadata, ChunkSamplingParams, ChunkStatus
from vllm.worker.worker import _pad_to_max

class ChunkRunner:
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def set_self_model_config(self, model: str) -> None:
        model_config = ModelConfig(model = model, tokenizer = None, tokenizer_mode = 'auto', 
                                   trust_remote_code = False, download_dir = None, use_np_weights = False,
                                   use_dummy_weights = False, dtype = 'auto', seed = 0)
        self.model_config = model_config
    
    def set_self_chunkworker(self, chunk_size: int, chunk_num: int) -> None:
        chunk_worker = ChunkWorker(chunk_size = chunk_size, chunk_num = chunk_num, 
                                   model_config = self.model_config)
        self.chunk_worker = chunk_worker
    
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
    
    def _add_requests_to_worker(self) -> None:
        #for prompt_token_ids in self.requests:
        cold_start_token_ids = [random.randint(0, 100) for _ in range(self.chunk_worker.chunk_size)]
        cold_start_sampling_params = ChunkSamplingParams(temperature = 0, top_p = 1.0, top_k = -1)
        self.chunk_worker.add_requests(prompt_token_ids = cold_start_token_ids, 
                                       sampling_params = cold_start_sampling_params)
        prompt_lens = [37, 45, 43, 43, 36, 36, 42, 41, 38, 45, 42, 40, 45, 43, 42, 45, 37, 38, 40, 40, 40, 45, 39, 
                       42, 44, 44, 43, 37, 41, 45, 45, 44, 38, 37, 45, 42, 44, 43, 42, 38, 38, 39, 45, 44, 35, 42, 
                       38, 35, 41, 35]
        for prompt_len in prompt_lens:
            sampling_params = ChunkSamplingParams(temperature = 0, top_p = 1.0, top_k = -1)
            #self.chunk_worker.add_requests(prompt_token_ids = prompt_token_ids, sampling_params = sampling_params)
            dummy_prompt_token_ids = [random.randint(0, 100) for _ in range(prompt_len)]
            self.chunk_worker.add_requests(prompt_token_ids = dummy_prompt_token_ids, sampling_params = sampling_params)
    
    def _start_worker(self) -> None:
        self._add_requests_to_worker()
        self.chunk_worker.set_job_sequences()
        self.chunk_worker.set_job_chunks()

    def _prepare_model_inputs(self, 
                              chunk: Chunk) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, List[Tuple[int, int, int]]]]:
        chunk_size = self.chunk_worker.chunk_size
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
            sequence = self.chunk_worker.job_sequences[seq_id]
            if kv_cache_num == 0:
                prompt_len = chunk.seqs_to_lens.get(seq_id)
                kv_cache_ids.setdefault(prompt_len, [])
            else:
                for chunk_id, used_token_num in sequence.chunks_to_prompts.items():
                    if chunk_id == chunk.chunk_id:
                        break
                    else:
                        block_id = self.chunk_worker.job_chunks[chunk_id].cache_block_id
                        st = 0
                        for a_seq_id, slice_token_num in self.chunk_worker.job_chunks[chunk_id].seqs_to_lens.items():
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

        for chunk in self.chunk_worker.job_chunks:
            chunk.chunk_status = ChunkStatus.RUNNING
            input_tokens_tensor, input_positions_tensor, kv_cache_ids = self._prepare_model_inputs(chunk)
            chunkinputmetadata = ChunkInputMetadata(prompt_lens = chunk.prompt_lens, 
                                                    kv_prefixs = chunk.kv_prefixs,
                                                    kv_prefixs_blocks = kv_cache_ids, 
                                                    kv_block = chunk.cache_block_id)
            output = self._execute_model(
                inputs = input_tokens_tensor,
                inputs_positions = input_positions_tensor,
                kv_cache = self.chunk_worker.kv_cache,
                chunkmetadata = chunkinputmetadata
            )
            st = 0
            idxs: List[int] = []
            sampling_params: List[ChunkSamplingParams] = []
            do_sampling: List[str] = []
            for seq_id, prompt_len in chunk.seqs_to_lens.items():
                ed = st + prompt_len
                self.chunk_worker.job_sequences[seq_id].update_count(prompt_len)
                if self.chunk_worker.job_sequences[seq_id].is_full():
                    idxs.append(ed - 1)
                    do_sampling.append(seq_id)
                    sampling_params.append(self.chunk_worker.job_sequences[seq_id].sampling_params)
                #self.chunk_worker.job_sequences[seq_id].outputs.append(output[st: ed])
                st = ed
                #self.chunk_worker.job_sequences[seq_id].add_start_and_end_time(st = start_time, ed = end_time)
                #st = ed
            output = output[idxs]
            output_token_list, logprobs = self.chunk_worker._execute_sampler(logits = output, 
                                                                             sampling_params = sampling_params)
            end_time = time.time()
            for i, id in enumerate(do_sampling):
                self.chunk_worker.job_sequences[id].add_first_token_id(output_token_list[i])
                self.chunk_worker.job_sequences[id].add_first_token_logprob(logprobs[i])
                self.chunk_worker.job_sequences[id].set_end_time(end_time)
            
        self.chunk_worker.reduce_outputs()
        #self.chunk_worker.generate_first_token_id()
        #self.chunk_worker.generate_first_token_str(tokenizer = self.tokenizer)
     
    @torch.inference_mode()
    def _execute_model(self, 
                       inputs: torch.Tensor, 
                       inputs_positions: torch.Tensor, 
                       kv_cache: List[Tuple[torch.Tensor, torch.Tensor]], 
                       chunkmetadata: ChunkInputMetadata) -> torch.Tensor: #Tuple[torch.Tensor, float, float]:
        #start_time = time.time()
        #print(inputs.shape)
        #print(inputs_positions.shape)
        output = self.chunk_worker.model(
            input_ids = inputs,
            positions = inputs_positions,
            kv_caches = kv_cache,
            cache_events = None,
            chunkinputmetadata = chunkmetadata
        )
        #end_time = time.time()
        return output #(output, start_time, end_time)