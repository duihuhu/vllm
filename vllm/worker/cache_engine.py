"""CacheEngine class for managing the KV cache."""
from typing import Dict, List, Tuple
import sys
import pickle
import torch

from vllm import cache_ops
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import in_wsl

from vllm import mem_ops
import numpy as np
from vllm.engine.plasma_client import plasma_client
from vllm.worker.object_manager.object_info import ObjectInfo
# import ctypes

logger = init_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor]

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
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_heads(parallel_config)
        self.dtype = model_config.dtype

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        # Initialize the cache.
        self.gpu_cache = self.allocate_gpu_cache()
        self.cpu_cache = self.allocate_cpu_cache()

        self.object_cache = self.allocate_object_cache()
                
        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        self.events = [torch.cuda.Event() for _ in range(self.num_layers)]

    def get_key_block_shape(self) -> Tuple[int, int, int, int]:
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        x = 16 // element_size
        return (
            self.num_heads,
            self.head_size // x,
            self.block_size,
            x,
        )

    def get_value_block_shape(self) -> Tuple[int, int, int]:
        return (
            self.num_heads,
            self.head_size,
            self.block_size,
        )

    def allocate_gpu_cache(self) -> List[KVCache]:
        gpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_gpu_blocks, *key_block_shape),
                dtype=self.dtype,
                device="cuda",
            )
            value_blocks = torch.empty(
                size=(self.num_gpu_blocks, *value_block_shape),
                dtype=self.dtype,
                device="cuda",
            )
            gpu_cache.append((key_blocks, value_blocks))
        return gpu_cache

    def allocate_cpu_cache(self) -> List[KVCache]:
        cpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        pin_memory = not in_wsl()
        if not pin_memory:
            # Pinning memory in WSL is not supported.
            # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
            logger.warning("Using 'pin_memory=False' as WSL is detected. "
                           "This may slow down the performance.")
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_cpu_blocks, *key_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
            )
            value_blocks = torch.empty(
                size=(self.num_cpu_blocks, *value_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
            )
            cpu_cache.append((key_blocks, value_blocks))
        return cpu_cache

    def calculate_object_size(self):
        return self.num_heads * self.head_size * self.block_size * _get_dtype_size(self.dtype)
    
    def allocate_object_cache(self) -> List[KVCache]:
        return 
    
    def _swap(
        self,
        src: List[KVCache],
        dst: List[KVCache],
        src_to_dst: Dict[int, int],
    ) -> None:
        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                src_key_cache, src_value_cache = src[i]
                dst_key_cache, dst_value_cache = dst[i]
                # Copy the key blocks.
                cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
                # Copy the value blocks.
                cache_ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)
        
                event = self.events[i]
                event.record(stream=self.cache_stream)

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)

    def _swap_prefilled(
        self,
        src: List[KVCache],
        dst: List[KVCache],
        src_to_dst: Dict[int, int],
    ) -> None:
        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                src_key_cache, src_value_cache = src[i]
                dst_key_cache, dst_value_cache = dst[i]
                # # Copy the key blocks.
                cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
                # Copy the value blocks.
                # cache_ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)
                
                event = self.events[i]
                event.record(stream=self.cache_stream)
                
                # mem_ops.print_blocks(src_value_cache, dst_value_cache, src_to_dst)
                # print("dst_value_cache: ", dst_value_cache)
                ##
                # print("element_size " ,src_key_cache.element_size(), src_key_cache[0].numel(), src_key_cache[0].shape, src_key_cache[0].dtype)
                
                # print(dst_value_cache[src_to_dst[0]])
                # for ks, ds in src_to_dst.items():
                #     print("ks, ds : ", dst_value_cache[ds].shape)
                # print(src_to_dst)
                # print(src_key_cache)
                # dst_value_cache_address = hex(id(dst_value_cache))
                ###
                # print("cpu blocks: ", self.num_cpu_blocks)
                # print("gpu blocks: ", self.num_gpu_blocks)
                # print("get_key_block_shape: ", self.get_key_block_shape())
                # print("get_value_block_shape: ", self.get_value_block_shape())
                
                # dst_value_cache_address = dst_value_cache.numpy().__array_interface__["data"][0]
                # print("dst_key_cache_address: ", hex(dst_value_cache_address))

    def _swap_out_prefilled_to_plasma(
       self,
        src: List[KVCache],
        src_to_dst: Dict[int, List[ObjectInfo]],
        rank) -> None:

        key_block_size_in_bytes = src[0][0].element_size() * src[0][0][0].numel()
        value_block_size_in_bytes = src[0][1].element_size() * src[0][1][0].numel()
        
        key_layer_object_address_lists = []
        for key, value in src_to_dst.items():
            print(value)
            object_info = value[rank].object_ids
            key_object_address_lists = []
            for object_id in object_info:
                obj = plasma_client.create(object_id, key_block_size_in_bytes)
                print("object and address" , obj, obj.address, rank)
                key_object_address_lists.append(obj.address)
            key_layer_object_address_lists.append(key_object_address_lists)
            
        # ##allocate key, value to objects and com by layer, lack swap value
        # key_layer_object_swap_lists = []
        # key_layer_object_address_lists = []
        # key_buf2obj = {}
        # for i in range(self.num_layers):
        #     key_object_swap_lists = []
        #     key_object_address_lists = []
        #     for key, value in src_to_dst.items():
        #         obj_id = plasma_client.allocate_object_id()
        #         obj = plasma_client.create(obj_id, key_block_size_in_bytes)
        #         key_object_swap_lists.append(obj)
        #         key_object_address_lists.append(obj.address)
        #         key_buf2obj[obj.address] = obj_id
        #     key_layer_object_swap_lists.append(key_object_swap_lists)
        #     key_layer_object_address_lists.append(key_object_address_lists)
            

        # with torch.cuda.stream(self.cache_stream):
        #     for i in range(self.num_layers):
        #         src_key_cache, src_value_cache = src[i]
        #         # dst_key_object = object_swap_lists[i]
        #         cache_ops.swap_blocks_to_object(src_key_cache, key_layer_object_address_lists[i], src_to_dst)
                
        # #seal object after swap data
        # for object_address_lists in key_layer_object_address_lists:
        #     for addr in object_address_lists:
        #         plasma_client.seal(key_buf2obj[addr])
        #         # buffer = plasma_client.get_buffers(buf2obj[addr])
                    
        #         # self.client.create(object_id, object_size)
        #         # memory_buffer = np.frombuffer(self.client.create(object_id, object_size), dtype=self.dtype)
        #         # print("src_key_cache, memory_buffer ", len(src_key_cache), len(memory_buffer))

        return
    
    def swap_out_prefilled(self, src_to_dst: Dict[int, int]) -> None:
        self._swap_prefilled(self.gpu_cache, self.cpu_cache, src_to_dst)

    def swap_out_prefilled_to_plasma(self, src_to_dst: Dict[int, List[ObjectInfo]], rank) -> None:
        self._swap_out_prefilled_to_plasma(self.gpu_cache, src_to_dst, rank)
    
    
    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        key_caches = [key_cache for key_cache, _ in self.gpu_cache]
        value_caches = [value_cache for _, value_cache in self.gpu_cache]
        # NOTE(woosuk): This operation implicitly synchronizes the CPU and GPU.
        cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)
    
    @staticmethod
    def get_cache_block_size(
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        dtype_size = _get_dtype_size(model_config.dtype)
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
