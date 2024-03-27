"""CacheEngine class for managing the KV cache."""
from typing import Dict, List, Tuple

import torch

from vllm._C import cache_ops


from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import in_wsl, STR_DTYPE_TO_TORCH_DTYPE

#todo list
# from vllm._C.gpu_ops import copy_blocks_in_layer, SendRequestRemote, RecvRequestRemote, SendBlocksRemote, RecvBlocksRemote

from vllm._C import gpu_ops
#hucc
# from torch.cuda import current_device, Stream, Event, stream

from vllm.core.kv_trans_scheduler import TransferTaskMeta


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
        request_id_size: int = 32,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        #hucc
        self.cache_size_per_block = self.block_size *self.num_heads * self.head_size * _get_dtype_size(self.dtype)
        self.request_id_size = request_id_size
        
        # Initialize the cache.
        self.gpu_cache = self.allocate_gpu_cache()
        self.cpu_cache = self.allocate_cpu_cache()

        #hucc Initialize the stream for caching operations
        self.swap_in_stream = torch.cuda.Stream(device=torch.cuda.current_device())
        self.swap_out_stream = torch.cuda.Stream(device=torch.cuda.current_device())
    
        
        #Initizlize the events for stream synchronization
        self.swap_in_events: Dict[str, torch.cuda.Event] = {}
        self.swap_out_events: Dict[str, torch.cuda.Event] = {}
        #send方在一个channel对应的Stream中只能有一个未完成时间，可能是传请求时间也可能是传数据时间
        #todo list FSM
        #send放在一个channel对应的Stream中可以有多个未完成时间，传请求和传数据绑定在一个事件中
        
        #request_id to request tensor
        #channel to request tensor
        
        self.send_streams: Dict[str, torch.cuda.Stream] = {}
        self.send_events: Dict[str, Tuple[str, torch.cuda.Event]] = {}
        self.send_waiting_request_ids: Dict[str, torch.Tensor] = {}
        
        self.recv_streams: Dict[str, torch.cuda.Stream] = {}
        self.recv_events: Dict[str, List[Tuple[str, torch.cuda.Event]]]
        self.recv_waiting_request_ids: Dict[str, torch.Tensor] = {}
        
        # Initialize the stream for caching operations.
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
    #hucc
    #for request id: send gpu->gpu , copy request id from gpu to cpu 
    def get_request_id_from_tensor(self, channel: str, device_tensor: torch.Tensor) -> str:
        with torch.cuda.stream(self.remote_send_streams[channel]):
            cpu_tensor = torch.ones(size=(self.request_id_size,), dtype=torch.uint8)
            cpu_tensor = device_tensor
            data_int = cpu_tensor.tolist()
        return ''.join([hex(data)[2:] for data in data_int])

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
                cache_ops.swap_blocks(src_value_cache, dst_value_cache,
                                      src_to_dst)
                event = self.events[i]
                event.record(stream=self.cache_stream)

    # def swap_in(self, src_to_dst: Dict[int, int]) -> None:
    #     self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)

    # def swap_out(self, src_to_dst: Dict[int, int]) -> None:
    #     self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)

    #hucc
    def swap_in(self, src_to_dst: Dict[int, int], key: str) -> None:
        with torch.cuda.stream(self.swap_in_stream):
            gpu_ops.copy_blocks_in_layer(self.gpu_cache, self.cpu_cache, src_to_dst, self.cache_size_per_block, True)
            event = torch.cuda.Event()
            event.record()
        self.swap_in_events[key] = event

    #todo  share one stream or two stream
    def swap_out(self, src_to_dst: Dict[int, int], key: str) -> None:
        with torch.cuda.stream(self.swap_in_stream):
            gpu_ops.copy_blocks_in_layer(self.cpu_cache, self.gpu_cache, src_to_dst, self.cache_size_per_block, False)
            event = torch.cuda.Event()
            event.record()
        self.swap_in_events[key] = event

    # pull语义, 由send方法调用
    def recv_request_id(self, channel: str, opposite_rank: int) -> str:
        if channel not in self.recv_streams:
            self.recv_streams[channel] = torch.cuda.Stream(device=torch.cuda.current_device())
            
        with torch.cuda.stream(self.recv_streams[channel]):
            tensor_of_request_id = torch.zeros(size=(self.request_id_size,),
                                               dtype=torch.uint8).cuda()
            gpu_ops.RecvRequestRemote(tensor_of_request_id.data_ptr(), self.request_id_size, opposite_rank)
            self.recv_waiting_request_ids[channel] = tensor_of_request_id
            event = torch.cuda.Event()
            event.record()
        self.recv_events[channel] = (None, event)
        
    def recv_blocks(self, channel: str, request_id: str, src_blocks: List[int], opposite_rank: int) -> None:      
        if channel not in self.recv_streams:
            self.recv_streams[channel] = torch.cuda.Stream(device=torch.cuda.current_device())
        
        with torch.cuda.stream(self.recv_streams[channel]):
            gpu_ops.RecvBlocksRemote(self.gpu_cache, src_blocks, self.cache_size_per_block, opposite_rank)
            event = torch.cuda.Event()
            event.record()
        self.recv_events[channel] = (request_id, event)

    def send_blocks(self, channel: str, request_id: str, dst_blocks: List[int], opposite_rank: int) -> str: 
        if channel not in self.send_streams:
            self.send_streams[channel] = torch.cuda.Stream(device=torch.cuda.current_device())
        
        with torch.cuda.stream(self.send_streams[channel]):
            tensor_of_request_id = torch.Tensor([int(data, 16) for data in list(request_id)]).byte().cuda()
            self.send_waiting_request_ids[request_id] = tensor_of_request_id
            gpu_ops.SendRequestRemote(tensor_of_request_id.data_ptr(), self.request_id_size, opposite_rank)
            gpu_ops.SendBlocksRemote(self.gpu_cache, dst_blocks, self.cache_size_per_block, opposite_rank)
            event = torch.cuda.Event()
            event.record() 
        if channel not in self.send_events:
            self.send_events[channel] = [(request_id, event)]
        else:
            self.send_events[channel].append((request_id, event))

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        key_caches = [key_cache for key_cache, _ in self.gpu_cache]
        value_caches = [value_cache for _, value_cache in self.gpu_cache]
        # NOTE(woosuk): This operation implicitly synchronizes the CPU and GPU.
        cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)

    def wait_for_swap_out_events(self, wait_for_swap_out: List[str]) -> None:
        for key in wait_for_swap_out:
            event = self.swap_out_events.pop(key)
            event.synchronize()
    
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
    
    #todo Tuple
    def check_send_finished_events(self) -> List[TransferTaskMeta]:
        #process send events
        send_finished_events: List[Tuple[str, List[int]]] = []
        send_blocks_finished: List[TransferTaskMeta] = []
        for channel, request_ids_and_events in self.send_events.items():
            for idx, (request_id, event) in enumerate(request_ids_and_events):
                if event.query():
                    send_blocks_finished.append(TransferTaskMeta(channel, request_id))
                    send_finished_events.append((channel, idx))
                    # request_tensor = self.remote_recv_waiting_request_ids[request_id]
                    del self.send_waiting_request_ids[request_id]
                else:
                    break
        for channel, idx in send_finished_events:
            self.send_events[channel].pop(idx)

        return send_blocks_finished
    
    def check_recv_finished_events(self) -> Tuple[List[TransferTaskMeta], List[TransferTaskMeta]]:
        #process recv events
        recv_finished_events: List[str] = []
        recv_request_id_finished: List[TransferTaskMeta] = []
        recv_blocks_finished: List[TransferTaskMeta] = []
        for channel, (request_id, event) in self.recv_events.items():
            if event.query():
                recv_finished_events.append(channel)
                #接收请求结束或发送数据结束
                if not request_id:
                    request_tensor = self.recv_waiting_request_ids[channel]
                    #提取请求request_id
                    finished_request_id = self.get_request_id_from_tensor(channel, request_tensor)
                    recv_request_id_finished.append(TransferTaskMeta(channel, finished_request_id))
                    #删除request_tensor
                    del self.recv_waiting_request_ids[channel]
                else:
                    recv_blocks_finished.append(TransferTaskMeta(channel,request_id))
        # release recv events
        for channel in recv_finished_events:
            self.recv_events.pop(channel)
            
        return recv_request_id_finished, recv_blocks_finished
        
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
