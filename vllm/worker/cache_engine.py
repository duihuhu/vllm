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
import ctypes
import time
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
                cache_ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)
                
                event = self.events[i]
                event.record(stream=self.cache_stream)

    def _swap_out_prefilled_to_plasma(
       self,
        src: List[KVCache],
        src_to_dst: Dict[int, List[ObjectInfo]],
        rank) -> None:
        # print("_swap_out_prefilled_to_plasma rank ", rank, rank % self.parallel_config.tensor_parallel_size)
        rank = rank % self.parallel_config.tensor_parallel_size
        key_block_size_in_bytes = src[0][0].element_size() * src[0][0][0].numel()
        value_block_size_in_bytes = src[0][1].element_size() * src[0][1][0].numel()
        
        print("value_block_size_in_bytes ", key_block_size_in_bytes, value_block_size_in_bytes)
        # start_create_prefilled_object = time.time()
        #by gpu block num compose
        # key_layer_object_address = []
        # for key, value in src_to_dst.items():
        #     # object_info = value[rank].object_ids
        #     for gpu_block_num, object_info in value[rank].items():
        #         key_object_address = []
        #         for object_id in object_info.object_ids:
        #             obj = plasma_client.create(object_id, key_block_size_in_bytes)
        #             key_object_address.append(obj.address)
        #         key_layer_object_address.append(key_object_address)
        
        #by layers compose
        key_layer_objects_address = []
        value_layer_objects_address = []
        
        for i in range(self.num_layers):
            key_objects_address = []
            value_objects_address = []
            
            for key, obj_info in src_to_dst.items():
                key_object_info = (obj_info[rank].object_ids)[0]
                value_object_info = (obj_info[rank].object_ids)[1]
                key_obj = plasma_client.create(key_object_info[i], key_block_size_in_bytes)
                key_objects_address.append(key_obj.address)
                value_obj = plasma_client.create(value_object_info[i], value_block_size_in_bytes)
                value_objects_address.append(value_obj.address)
                
            key_layer_objects_address.append(key_objects_address)
            value_layer_objects_address.append(value_objects_address)
        src_to_dst_copy = {}
        for key, _ in src_to_dst.items():
            src_to_dst_copy[key] = 0
        # end_create_prefilled_object = time.time()
        # print("start_create_prefilled_object, end create_prefilled_object time ", start_create_prefilled_object, end_create_prefilled_object, rank)

        # ##allocate key, value to objects and com by layer, lack swap value, (init version)
        # key_layer_objects_swap = []
        # key_layer_objects_address = []
        # key_buf2obj = {}
        # for i in range(self.num_layers):
        #     key_objects_swap = []
        #     key_objects_address = []
        #     for key, value in src_to_dst.items():
        #         obj_id = plasma_client.allocate_object_id()
        #         obj = plasma_client.create(obj_id, key_block_size_in_bytes)
        #         key_objects_swap.append(obj)
        #         key_objects_address.append(obj.address)
        #         key_buf2obj[obj.address] = obj_id
        #     key_layer_objects_swap.append(key_objects_swap)
        #     key_layer_objects_address.append(key_objects_address)

        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                src_key_cache, src_value_cache = src[i]
                cache_ops.swap_blocks_to_object(src_key_cache, key_layer_objects_address[i], src_to_dst_copy, 0)
                cache_ops.swap_blocks_to_object(src_value_cache, value_layer_objects_address[i], src_to_dst_copy, 0)
                event = self.events[i]
                event.record(stream=self.cache_stream)
                # print("swap out layer i, key ", i, key_layer_objects_address[i])
                # print("swap out layer i, value ", i, value_layer_objects_address[i])

        # start_seal_prefilled_object = time.time()
        for key, obj_info in src_to_dst.items():
            key_object_info = (obj_info[rank].object_ids)[0]
            value_object_info = (obj_info[rank].object_ids)[1]
            for key_addr, value_addr in zip(key_object_info, value_object_info):            
                plasma_client.seal(key_addr)
                plasma_client.seal(value_addr)
        # end_seal_prefilled_object = time.time()
        # print("start_seal_prefilled_object, end_seal_prefilled_object time ", start_seal_prefilled_object, end_seal_prefilled_object, rank)
        #seal object after swap data
        # for object_address_lists in key_layer_object_address_lists:
        #     for addr in object_address_lists:
        #         plasma_client.seal(key_buf2obj[addr])
        
    
    def swap_out_prefilled(self, src_to_dst: Dict[int, int]) -> None:
        self._swap_prefilled(self.gpu_cache, self.cpu_cache, src_to_dst)

    def swap_out_prefilled_to_plasma(self, src_to_dst: Dict[int, List[ObjectInfo]], rank) -> None:
        self._swap_out_prefilled_to_plasma(self.gpu_cache, src_to_dst, rank)
    
    def _swap_in_prefilled_from_plasma(self, src: List[KVCache], src_to_dst: Dict[int, List[ObjectInfo]], rank, kv_data) -> None:
        rank = rank % self.parallel_config.tensor_parallel_size
        # print("_swap_in_prefilled_to_plasma rank ", rank, rank % self.parallel_config.tensor_parallel_size)
        src_to_dst_copy = {}
        key_object_address = []
        value_object_address = []
        
        key_socket_object_address = []
        value_socket_object_address = []
        
                
        key_socket_object_content = []
        value_socket_object_content = []
        # for key, obj_info in src_to_dst.items():
        #     src_to_dst_copy[key] = 0
        #     key_obj_info = (obj_info[rank].object_ids)[0]
        #     value_obj_info = (obj_info[rank].object_ids)[1]
        #     key_obj_buf = plasma_client.get_buffers(key_obj_info)
        #     value_obj_buf = plasma_client.get_buffers(value_obj_info)
        #     key_obj_addr = []
        #     value_obj_addr = []
        #     for k_addr, v_addr in zip(key_obj_buf, value_obj_buf):
        #         key_obj_addr.append(k_addr.address)
        #         value_obj_addr.append(v_addr.address)
        #     key_object_address.append(key_obj_addr)
        #     value_object_address.append(value_obj_addr)

        for key, obj_info in src_to_dst.items():
            src_to_dst_copy[key] = 0
            key_obj_info = (obj_info[rank].object_ids)[0]
            value_obj_info = (obj_info[rank].object_ids)[1]
            key_obj_buf = plasma_client.get_buffers(key_obj_info)
            value_obj_buf = plasma_client.get_buffers(value_obj_info)
            key_obj_addr = []
            value_obj_addr = []
            key_socket_obj_addr = []
            value_socket_obj_addr = []
            key_socket_content = []
            value_socket_content = []
            for k_addr, v_addr, key_obj, value_obj in zip(key_obj_buf, value_obj_buf, key_obj_info, value_obj_info):
                key_obj_addr.append(k_addr.address)
                value_obj_addr.append(v_addr.address)
                # key_value = kv_data[key_obj.binary().hex()]
                # v_value = kv_data[value_obj.binary().hex()]
                # key_value = ctypes.addressof(ctypes.c_char.from_buffer_copy(kv_data[key_obj.binary().hex()]))
                # print("key value ", key_value)
                # data_at_address = ctypes.string_at(key_value, 10)
                
                # key_value = (ctypes.c_char * len(kv_data[key_obj.binary().hex()])).from_buffer_copy(kv_data[key_obj.binary().hex()])
                # address = ctypes.addressof(key_value)
                key_value_buffer = memoryview(kv_data[key_obj.binary().hex()])
                key_value = ctypes.addressof(key_value_buffer)
                
                v_value = ctypes.addressof(ctypes.c_char.from_buffer_copy(kv_data[value_obj.binary().hex()]))

                key_socket_obj_addr.append(key_value)
                value_socket_obj_addr.append(v_value)
                # key_socket_content.append(key_value)
                # value_socket_content.append(v_value)
                
            key_object_address.append(key_obj_addr)
            value_object_address.append(value_obj_addr)
            key_socket_object_address.append(key_socket_obj_addr)
            value_socket_object_address.append(value_socket_obj_addr)
            key_socket_object_content.append(key_socket_content)
            value_socket_object_content.append(value_socket_content)
            
        for k_obj, ks_obj in zip(key_obj_addr, key_socket_obj_addr):
            k_obj_ptr = ctypes.c_void_p(k_obj)
            k_obj_raw_data = ctypes.string_at(k_obj_ptr, 10)
            
            ks_obj_ptr = ctypes.c_void_p(ks_obj)
            ks_obj_raw_data = ctypes.string_at(ks_obj_ptr, 10)
            print("00: ", k_obj_raw_data)
            print("11: ", ks_obj_raw_data)
        # for key, obj_info in src_to_dst.items():
        #     src_to_dst_copy[key] = 0
        #     key_obj_info = (obj_info[rank].object_ids)[0]
        #     value_obj_info = (obj_info[rank].object_ids)[1]
        #     key_obj_addr = []
        #     value_obj_addr = []
        #     for key_obj, value_obj in zip(key_obj_info, value_obj_info):
        #         if kv_data.get(key_obj.binary().hex()):
        #             print("key " , id(kv_data[key_obj.binary().hex()]))
        #             key_obj_addr.append(id(kv_data[key_obj.binary().hex()]))
        #         if kv_data.get(value_obj.binary().hex()):
        #             print("value " , id(kv_data[value_obj.binary().hex()]))
        #             value_obj_addr.append(id(kv_data[value_obj.binary().hex()]))
        #     key_object_address.append(key_obj_addr)
        #     value_object_address.append(value_obj_addr)
        
        #by layers compose
        key_layer_objects_address = []
        value_layer_objects_address = []
        for i in range(self.num_layers):
            key_objects_address = []
            value_objects_address = []
            j = 0
            for key, obj_info in src_to_dst.items():                
                key_objects_address.append(key_object_address[j][i])
                value_objects_address.append(value_object_address[j][i])
                j = j + 1
            key_layer_objects_address.append(key_objects_address)
            value_layer_objects_address.append(value_objects_address)
            
        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                src_key_cache, src_value_cache = src[i]
                cache_ops.swap_blocks_to_object(src_key_cache, key_layer_objects_address[i], src_to_dst_copy, 1)
                cache_ops.swap_blocks_to_object(src_value_cache, value_layer_objects_address[i], src_to_dst_copy, 1)
                event = self.events[i]
                event.record(stream=self.cache_stream)
                # print("swap in layer i, key ", i, key_layer_objects_address[i])
                # print("swap in layer i, value ", i, value_layer_objects_address[i])
        return 
    
    def swap_in_prefilled_from_plasma(self, src_to_dst:  Dict[int, List[ObjectInfo]], rank, kv_data) -> None:
        self._swap_in_prefilled_from_plasma(self.gpu_cache, src_to_dst, rank, kv_data)
    
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
