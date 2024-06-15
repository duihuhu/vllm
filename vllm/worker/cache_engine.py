"""CacheEngine class for managing the KV cache."""
from typing import Dict, List, Tuple, Optional


import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, DeployConfig
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_pin_memory_available


from vllm._C import gpu_ops, ops

from vllm.core.kv_trans_scheduler import TransferTaskMeta

logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        deploy_config: DeployConfig,
        worker_rank: int,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.use_agg_block = deploy_config.use_agg_block

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        self.deploy_config = deploy_config
        
        self.worker_rank = worker_rank
        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        #hucc
        self.cache_size_per_block = self.block_size *self.num_heads * self.head_size * _get_dtype_size(self.dtype)
        self.cache_block_size =  self.num_layers * self.num_heads * self.head_size * self.block_size
        
        #Initizlize the events for stream synchronization
        self.swap_in_events: Dict[str, torch.cuda.Event] = {}
        self.swap_out_events: Dict[str, torch.cuda.Event] = {}
                
        # Get attention backend.
        self.attn_backend = get_attn_backend(model_config.dtype)

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(self.num_gpu_blocks, "cuda", self.use_agg_block)
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu", self.use_agg_block)

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
        use_agg_block: Optional[bool] = False
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        if use_agg_block:
            kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_heads, self.head_size, self.num_layers)
            for _ in range(self.num_gpu_blocks):
                kv_cache.append(
                    torch.empty(
                        kv_cache_shape,
                        dtype=self.dtype,
                        pin_memory=pin_memory,
                        device=device))
        else:
            kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                num_blocks, self.block_size, self.num_heads, self.head_size, None)
            for _ in range(self.num_layers):
                kv_cache.append(
                    torch.empty(kv_cache_shape,
                                dtype=self.dtype,
                                pin_memory=pin_memory,
                                device=device))
        return kv_cache
    
    def get_tensor_for_caches_address(self, gpu: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_agg_block:
            key_caches = []
            value_caches = []
            if gpu is True:
                for cache_block in self.gpu_cache:
                    key_caches.append(cache_block[0])
                    value_caches.append(cache_block[1])
                key_caches_ptrs_tensor = ops.tensor_for_caches_addresses(key_caches)
                value_caches_ptrs_tensor = ops.tensor_for_caches_addresses(value_caches)
                return (key_caches_ptrs_tensor, value_caches_ptrs_tensor)
            else:
                for cache_block in self.cpu_cache:
                    key_caches.append(cache_block[0])
                    value_caches.append(cache_block[1])
                key_caches_ptrs_tensor = ops.tensor_for_caches_addresses(key_caches)
                value_caches_ptrs_tensor = ops.tensor_for_caches_addresses(value_caches)
                return (key_caches_ptrs_tensor, value_caches_ptrs_tensor)
        else:
            #print(f"In get_tensor_for_caches_address the use-agg-block is False")
            return (None, None)

    def get_blocks_address(self, gpu: bool) -> List[int]:
        if self.use_agg_block:
            key_caches = []
            if gpu is True:
                for cache_block in self.gpu_cache:
                    key_caches.append(cache_block[0])
                blocks_address = ops.tensor_for_blocks_address(key_caches)
                return blocks_address
            else:
                for cache_block in self.cpu_cache:
                    key_caches.append(cache_block[0])
                blocks_address = ops.tensor_for_blocks_address(key_caches)
                return blocks_address
        else:
            return None


    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def swap_by_agg2_in(self, src_to_dst: Dict[int, int]) -> None:
        block_size_in_bytes = self.gpu_cache[0].element_size() * self.gpu_cache[0].numel()
        for src, dst in src_to_dst.items():
            self.attn_backend.swap_blocks_agg2(self.cpu_cache[src], self.gpu_cache[dst], block_size_in_bytes)
			
    def swap_by_agg2_out(self, src_to_dst: Dict[int, int]) -> None:
        block_size_in_bytes = self.gpu_cache[0].element_size() * self.gpu_cache[0].numel()
        for src, dst in src_to_dst.items():
            self.attn_backend.swap_blocks_agg2(self.gpu_cache[src], self.cpu_cache[dst], block_size_in_bytes)


    def swap_by_agg(self,
                    src_addresses: Tuple[torch.Tensor, torch.Tensor],
                    dst_addresses: Tuple[torch.Tensor, torch.Tensor],
                    src_to_dst: Dict[int, int]) -> None:
        block_size_in_bytes = self.gpu_cache[0][0].element_size() * self.gpu_cache[0][0].numel()
        self.attn_backend.swap_blocks_agg(src_addresses, dst_addresses, src_to_dst, block_size_in_bytes)


    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)
		
    def copy_agg(self, kv_cache_addresses: Tuple[torch.Tensor, torch.Tensor],
                 src_to_dsts: Dict[int, List[int]]) -> None:
        num_layers = self.gpu_cache[0][0].shape[0]
        numel_per_layer = self.gpu_cache[0][0].stride(0)
        self.attn_backend.copy_blocks_agg(kv_cache_addresses, src_to_dsts, num_layers, numel_per_layer)

    def check_finished_events(self) -> Tuple[List[str], List[str]]:
        #swap in events
        swap_in_finished_req_ids: List[str] = []
        for key, event in self.swap_in_events.items():
            if event.query():
                swap_in_finished_req_ids.append(key)
            else:
                break
        for key in swap_in_finished_req_ids:
            self.swap_in_events.pop(key)
        swap_out_finished_req_ids: List[str] = []
        for key, event in self.swap_out_events.items():
            if event.query():
                swap_out_finished_req_ids.append(key)
            else:
                break
        for key in swap_out_finished_req_ids:
            self.swap_out_events.pop(key)
        return (swap_in_finished_req_ids, swap_out_finished_req_ids)

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        cache_dtype: str,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        if cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        dtype_size = _get_dtype_size(dtype)
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
