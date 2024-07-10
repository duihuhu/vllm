"""
    the main frame of RTC(relational tensor cache).
"""
from typing import Union, Tuple, Dict, List

import torch
from vllm.global_vars import ENABLE_NPU
if ENABLE_NPU:
    from torch_npu.npu import current_device
else:
    from torch.cuda import current_device
from vllm.sequence import SequenceData, Sequence
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata

from vllm.utils import Counter
from vllm.radix_tree_ys.radix_tree_manager import RadixTreeManager
from vllm.radix_tree_ys.radix_cache import TreeNode


class RtcEngine:
    def __init__(
        self, 
        block_size: int,
        gpu_usage_threshold: Tuple[float, float],
        cpu_usage_threshold: Tuple[float, float],
        max_swap_gpu_blocks_per_step: int,
        swap_group_size: int,
    ) -> None:
        """A tensor cache optimizing system for LLM inference
        
        Args: 
        """
        self.radix_tree_manager = RadixTreeManager(block_size=block_size)
        self.block_size = block_size
        # rtc hyperparameters
        # (lower_threshold, upper_threshold), when memory usage larger than upper_threshold,
        # scheduler will try to swap or free blocks to lower_threshold
        self.gpu_usage_threshold = gpu_usage_threshold
        self.cpu_usage_threshold = cpu_usage_threshold
        self.max_swap_gpu_blocks_per_step = max_swap_gpu_blocks_per_step # avoid buffer full
        self.swap_group_size = swap_group_size # grouped swap


    def inference_optimizing(
        self, 
        seq: Sequence, 
        inference_type: str = "match", 
        free_call_back = None) -> bool:
        """unified external entrance for inference optimizing in RTC
        
        Args:
            inference_type: "match" or "insert"
        """
        if inference_type == "match":
            return self.radix_tree_manager.match(seq)
        elif inference_type == "insert":
            return self.radix_tree_manager.insert(seq, free_call_back)
        else:
            raise ValueError(f'Invalid arg: inference_type={inference_type}')
        
    ##### Public API For Radix Tree ##### 
    def refresh_nodes_status(self, nodes: List[TreeNode]):
        self.radix_tree_manager.refresh_nodes_status(nodes)
        
    def get_num_nodes_can_swap_out(self) -> int:
        return self.radix_tree_manager.get_num_nodes_can_swap_out()
        
    def swap_out_nodes(self, num_nodes: int) -> List[TreeNode]:
        return self.radix_tree_manager.swap_out(num_nodes)
        
    def evict_nodes(self, num_nodes, device, evict_callback) -> int:
        return self.radix_tree_manager.evict(num_nodes, device, evict_callback)
    
    def get_num_nodes(self, device = None) -> int:
        return self.radix_tree_manager.get_num_nodes(device)

    @staticmethod
    def check_using_cache_with_sampling_metadata(
            sampling_metadata: SamplingMetadata):
        seq_data_key: List[int] = list(sampling_metadata.seq_data.keys())
        return any(sampling_metadata.seq_data[key].cache_token_len > 0 for key in seq_data_key)

    @staticmethod
    def preprocess_input(input_ids: torch.Tensor,
                         input_positions: torch.Tensor,
                         input_metadata: InputMetadata,) -> [torch.Tensor, torch.Tensor]:
        """the xds_rtc preprocess the input for inference optimize when application preare input

        Args:
            input_ids: the inference input token
            input_positions: the position of inference input token
            input_metadata: the all meta data of input
        """
        seq_data_key: List[int] = list(input_metadata.seq_data.keys())
        device = current_device()
        pos: int = 0
        indices = []
        for i in range(len(seq_data_key)):
            seq_data: SequenceData = input_metadata.seq_data[seq_data_key[i]]
            start = seq_data.cache_token_len + pos
            end = seq_data.get_len() + pos
            indices.extend(range(start, end))
            pos += seq_data.get_len()
            input_metadata.prompt_lens[i] -= seq_data.cache_token_len
            input_metadata.num_prompt_tokens -= seq_data.cache_token_len
        indices = torch.tensor(indices, dtype=torch.long, device=device)
        input_metadata.slot_mapping = input_metadata.slot_mapping.index_select(0, indices)
        input_metadata.num_valid_tokens = input_metadata.slot_mapping.shape[0]
        return input_ids.index_select(0, indices), input_positions.index_select(0, indices)

    @staticmethod
    def preprocess_input_with_sampling_metadata(input_ids: torch.Tensor,
                                                input_positions: torch.Tensor,
                                                origin_slot_mapping: torch.Tensor,
                                                sampling_metadata: SamplingMetadata = None,
                                                ) -> [torch.Tensor, torch.Tensor]:
        seq_data_key: List[int] = list(sampling_metadata.seq_data.keys())
        device = current_device()
        pos: int = 0
        indices = []
        for i in range(len(seq_data_key)):
            seq_data: SequenceData = sampling_metadata.seq_data[seq_data_key[i]]
            start = seq_data.cache_token_len + pos
            end = seq_data.get_len() + pos
            indices.extend(range(start, end))
            pos += seq_data.get_len()
            sampling_metadata.prompt_lens[i] -= seq_data.cache_token_len
            sampling_metadata.num_prompt_tokens -= seq_data.cache_token_len
        indices = torch.tensor(indices, dtype=torch.long, device=device)
        updated_slot_mapping = origin_slot_mapping.index_select(0, indices)
        return input_ids.index_select(0, indices), input_positions.index_select(0, indices), updated_slot_mapping

    @staticmethod
    def generate_prefill_attn_mask(input_metadata: InputMetadata,
                                   attn_mask: torch.Tensor, ) -> torch.Tensor:
        """the xds_rtc preprocess the attention mask tensor for multi round dialogue and so on
        
        Args:
            input_metadata: the all meta data of input
            attn_mask: the origin attention mask
        """
        seq_data_key: List[int] = list(input_metadata.seq_data.keys())
        device = current_device()
        indices = []
        pos: int = 0
        for key in seq_data_key:
            seq_data: SequenceData = input_metadata.seq_data[key]
            start = seq_data.cache_token_len + pos
            end = seq_data.get_len() + pos
            indices.extend(range(start, end))
            pos += seq_data.get_len()
        indices = torch.tensor(indices, dtype=torch.long, device=device)
        return attn_mask.index_select(0, indices)
    
    @staticmethod
    def splice_prefill_cache(key: torch.Tensor, value: torch.Tensor,
                             key_cache: torch.Tensor, value_cache: torch.Tensor,
                             input_metadata: InputMetadata, ) -> [torch.Tensor, torch.Tensor]:
        """splcie the hit kv cache because of the prefill not support the discontinuous memory
        
        Args:
            key: the k tensor of projection
            value: the v tensor of projection
            key_cache: the ptr of whole application's attention key tensor cache
            value_cache: the ptr of whole application's attention value tensor cache
            input_metadata: the all meta data of input
        """
        seq_data_key: List[int] = list(input_metadata.seq_data.keys())
        key_new = key
        value_new = value
        start: int = 0
        #TODO: This has a problem on batch prefill, so now not support batch prefill
        for key_id in seq_data_key:
            seq_data: SequenceData = input_metadata.seq_data[key_id]
            block_idx: List[int] = seq_data.prefix_block_ids
            idx_end = seq_data.cache_token_len % input_metadata.block_size
            end = seq_data.get_len() - seq_data.cache_token_len
            if ENABLE_NPU:
                for idx in reversed(block_idx):
                    if idx == block_idx[-1] and idx_end > 0:
                        key_new = torch.cat((key_cache[idx].permute(1, 0, 2)[:idx_end], key_new), 0)
                        value_new = torch.cat((value_cache[idx].permute(1, 0, 2)[:idx_end], value_new), 0)
                    else:
                        key_new = torch.cat((key_cache[idx].permute(1, 0, 2), key_new), 0)
                        value_new = torch.cat((value_cache[idx].permute(1, 0, 2), value_new), 0)
            else:
                _, num_kv_heads, head_size, _ = value_cache.shape
                for idx in reversed(block_idx):
                    if idx == block_idx[-1] and idx_end > 0:  # copy the valid tensor of block
                        key_new = torch.cat((key_cache[idx].permute(2, 0, 1, 3)[:idx_end].reshape(-1, num_kv_heads,
                                                                                              head_size), key_new),
                                            0)
                        value_new = torch.cat((value_cache[idx].permute(2, 0, 1)[:idx_end], value_new), 0)
                    else:  # copy the whole of block
                        key_new = torch.cat((key_cache[idx].permute(2, 0, 1, 3).reshape(-1, num_kv_heads,
                                                                                        head_size), key_new), 0)
                        value_new = torch.cat((value_cache[idx].permute(2, 0, 1), value_new), 0)
            start += end
        return key_new, value_new

    @staticmethod
    def get_prefill_cache(input_metadata: InputMetadata) -> [torch.Tensor, torch.Tensor]:
        """
        get cached block table and cached context length
        Args:
            input_metadata: metadata of input
        """
        seq_data_keys: List[int] = list(input_metadata.seq_data.keys())
        device = current_device()
        seq_data_list: List[SequenceData] = [input_metadata.seq_data[key_id] for key_id in seq_data_keys]
        block_idx: List[List[int]] = [seq_data.prefix_block_ids for seq_data in seq_data_list]
        context_len: List[int] = [seq_data.cache_token_len for seq_data in seq_data_list]
        return torch.tensor(block_idx, device=device, dtype=torch.int32), \
            torch.tensor(context_len, device=device, dtype=torch.int32)

    @staticmethod
    def get_prefill_cache_with_sampling_metadata(sampling_metadata: SamplingMetadata) -> torch.Tensor:
        seq_data_keys: List[int] = list(sampling_metadata.seq_data.keys())
        device = current_device()
        seq_data_list: List[SequenceData] = [sampling_metadata.seq_data[key_id] for key_id in seq_data_keys]
        context_len: List[int] = [seq_data.cache_token_len for seq_data in seq_data_list]
        return torch.tensor(context_len, device=device, dtype=torch.int32)