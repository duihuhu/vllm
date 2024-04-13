"""CacheEngine class for managing the KV cache."""
from typing import Dict, List, Tuple


import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_pin_memory_available


from vllm._C import gpu_ops

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
        self.recv_events: Dict[str, List[Tuple[str, torch.cuda.Event]]] = {}
        self.recv_waiting_request_ids: Dict[str, torch.Tensor] = {}
        
        # Get attention backend.
        self.attn_backend = get_attn_backend(model_config.dtype)

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(self.num_gpu_blocks, "cuda")
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    #hucc
    #for request id: send gpu->gpu , copy request id from gpu to cpu 
    def get_request_id_from_tensor(self, device_tensor: torch.Tensor) -> str:
        cpu_tensor = torch.ones(size=(self.request_id_size,), dtype=torch.uint8)
        cpu_tensor = device_tensor
        data_int = cpu_tensor.tolist()
        return ''.join([hex(data)[2:] for data in data_int])
    
    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            kv_cache.append(
                torch.empty(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
        return kv_cache

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    #hucc
    def swap_in_in_layer(self, src_to_dst: Dict[int, int], key: str) -> None:
        cpu_cache = [(kv_cache[0], kv_cache[1]) for kv_cache in self.cpu_cache]
        gpu_cache = [(kv_cache[0], kv_cache[1]) for kv_cache in self.gpu_cache]
        with torch.cuda.stream(self.swap_in_stream):
            gpu_ops.copy_blocks_in_layer(gpu_cache, cpu_cache, src_to_dst, self.cache_size_per_block, True)
            event = torch.cuda.Event()
            event.record()
        self.swap_in_events[key] = event

    #todo  share one stream or two stream
    def swap_out_in_layer(self, src_to_dst: Dict[int, int], key: str) -> None:
        cpu_cache = [(kv_cache[0], kv_cache[1]) for kv_cache in self.cpu_cache]
        gpu_cache = [(kv_cache[0], kv_cache[1]) for kv_cache in self.gpu_cache]
        with torch.cuda.stream(self.swap_in_stream):
            gpu_ops.copy_blocks_in_layer(cpu_cache, gpu_cache, src_to_dst, self.cache_size_per_block, False)
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
        print("recv_blocks ", len(src_blocks))
        gpu_cache = [(kv_cache[0], kv_cache[1]) for kv_cache in self.gpu_cache]
        with torch.cuda.stream(self.recv_streams[channel]):
            gpu_ops.RecvBlocksRemote(gpu_cache, src_blocks, self.cache_size_per_block, opposite_rank)
            event = torch.cuda.Event()
            event.record()
        self.recv_events[channel] = (request_id, event)

    def send_blocks(self, channel: str, request_id: str, dst_blocks: List[int], opposite_rank: int) -> str: 
        if channel not in self.send_streams:
            self.send_streams[channel] = torch.cuda.Stream(device=torch.cuda.current_device())
        print("send_blocks ", len(dst_blocks))
        gpu_cache = [(kv_cache[0], kv_cache[1]) for kv_cache in self.gpu_cache]
        with torch.cuda.stream(self.send_streams[channel]):
            tensor_of_request_id = torch.Tensor([int(data, 16) for data in list(request_id)]).byte().cuda()
            self.send_waiting_request_ids[request_id] = tensor_of_request_id
            gpu_ops.SendRequestRemote(tensor_of_request_id.data_ptr(), self.request_id_size, opposite_rank)
            gpu_ops.SendBlocksRemote(gpu_cache, dst_blocks, self.cache_size_per_block, opposite_rank)
            event = torch.cuda.Event()
            event.record() 
        if channel not in self.send_events:
            self.send_events[channel] = [(request_id, event)]
        else:
            self.send_events[channel].append((request_id, event))
            
    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)


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
                    finished_request_id = self.get_request_id_from_tensor(request_tensor)
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
