from transformers import PreTrainedTokenizerBase
import json
from typing import List, Dict, Tuple
import torch

from vllm.config import ModelConfig
from vllm.chunked.chunkworker import ChunkWorker
from vllm.chunked.chunk import Chunk, ChunkInputMetadata, ChunkSamplingParams, ChunkStatus
from vllm.worker.worker import _pad_to_alignment

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
        for prompt_token_ids in self.requests:
            sampling_params = ChunkSamplingParams(temperature = 0.8, top_p = 1.0, top_k = -1)
            self.chunk_worker.add_requests(prompt_token_ids = prompt_token_ids, sampling_params = sampling_params)
    
    def _start_worker(self) -> None:
        self._add_requests_to_worker()
        self.chunk_worker.set_job_sequences()
        self.chunk_worker.set_job_chunks()

    def _prepare_model_inputs(self, 
                              chunk: Chunk) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, List[Tuple[int, int, int]]]]:
        input_tokens = chunk.prompt_token_ids
        input_positions: List[int] = []
        for i, slice_length in enumerate(chunk.prompt_lens):
            slice_positions: List[int] = list(range(slice_length))
            slice_positions = [x + chunk.kv_prefixs[i] for x in slice_positions]
            input_positions.extend(slice_positions)
        input_tokens = _pad_to_alignment(input_tokens, multiple_of = 8)
        input_tokens_tensor = torch.cuda.LongTensor(input_tokens)
        input_positions = _pad_to_alignment(input_positions, multiple_of = 8)
        input_positions_tensor = torch.cuda.LongTensor(input_positions)
        kv_cache_ids: Dict[int, List[Tuple[int, int, int]]] = {} # prompt_len to (block, st, length)s
        for seq_id, kv_cache_num in chunk.seqs_to_prefixs.items():
            if kv_cache_num == 0:
                continue
            sequence = self.chunk_worker.job_sequences[seq_id]
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

        for chunk in self.chunk_worker.job_chunks:
            chunk.chunk_status = ChunkStatus.RUNNING
            input_tokens_tensor, input_positions_tensor, kv_cache_ids = self._prepare_model_inputs(chunk)
            chunkinputmetadata = ChunkInputMetadata(prompt_lens = chunk.prompt_lens, kv_prefixs = chunk.kv_prefixs,
                                                    kv_prefixs_blocks = kv_cache_ids, kv_block = chunk.cache_block_id)
            # add for debug
            print(f"chunk id: {chunk.chunk_id}")
            print(f"prompt lens: {chunk.prompt_lens}")
            print(f"seq2lens: {chunk.seqs_to_lens}")
            print(f"kv prefixs: {chunk.kv_prefixs}")
            print(f"kv prefixs blocks: {kv_cache_ids}")
            print(f"self cache block id: {chunk.cache_block_id}")
            '''output = self._execute_model(
                inputs = input_tokens_tensor,
                inputs_positions = input_positions_tensor,
                kv_cache = self.chunk_worker.kv_cache,
                chunkmetadata = chunkinputmetadata
            )
            for seq_id, prompt_len in chunk.seqs_to_lens.items():
                self.chunk_worker.job_sequences[seq_id].outputs.append(output[: prompt_len])
        
        self.chunk_worker.reduce_outputs()
        self.chunk_worker.generate_first_token_id()
        self.chunk_worker.generate_first_token_str(tokenizer = self.tokenizer)'''
     
    @torch.inference_mode()
    def _execute_model(self, inputs: torch.Tensor, inputs_positions: torch.Tensor, 
                       kv_cache: List[Tuple[torch.Tensor, torch.Tensor]], chunkmetadata: ChunkInputMetadata) -> torch.Tensor:
        output = self.chunk_worker.model(
            inputs,
            inputs_positions,
            kv_cache,
            chunkmetadata
        )
        return output