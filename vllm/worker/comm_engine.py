"""CacheEngine class for managing the KV cache."""
from typing import Dict, List, Tuple


import torch

from vllm.config import CacheConfig, ModelConfig, ParallelConfig, DeployConfig
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_pin_memory_available


from vllm._C import gpu_ops

from vllm.core.kv_trans_scheduler import TransferTaskMeta

logger = init_logger(__name__)

class CommEngine:
    """Manages the KV cache Addr.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches Addr. for nccl
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        deploy_config: DeployConfig,
        gpu_cache_addr: List[Tuple[List[int], List[int]]],
        request_id_size: int = 32,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size

        self.deploy_config = deploy_config
        
        self.gpu_cache_addr = gpu_cache_addr
        
        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        #hucc
        self.cache_size_per_block = self.block_size *self.num_heads * self.head_size * _get_dtype_size(self.dtype)
        self.request_id_size = request_id_size
    
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
        

    #hucc
    #for request id: send gpu->gpu , copy request id from gpu to cpu 
    def get_request_id_from_tensor(self, device_tensor: torch.Tensor) -> str:
        cpu_tensor = torch.ones(size=(self.request_id_size,), dtype=torch.uint8)
        cpu_tensor = device_tensor
        data_int = cpu_tensor.tolist()
        return ''.join([hex(data)[2:] for data in data_int])
    

    # pull语义, 由send方法调用
    def recv_request_id(self, channel: str, opposite_rank: int) -> str:
        if channel not in self.recv_streams:
            self.recv_streams[channel] = torch.cuda.Stream(device=torch.cuda.current_device())
            
        with torch.cuda.stream(self.recv_streams[channel]):
            tensor_of_request_id = torch.zeros(size=(self.request_id_size,),
                                               dtype=torch.uint8).cuda()
            gpu_ops.RecvRequest(tensor_of_request_id.data_ptr(), self.request_id_size, opposite_rank)
            self.recv_waiting_request_ids[channel] = tensor_of_request_id
            event = torch.cuda.Event()
            event.record()
        self.recv_events[channel] = (None, event)
        print("recv_request_id", channel, opposite_rank)
        
    def recv_blocks(self, channel: str, request_id: str, src_blocks: List[int], opposite_rank: int) -> None:      
        if channel not in self.recv_streams:
            self.recv_streams[channel] = torch.cuda.Stream(device=torch.cuda.current_device())
        # print("recv_blocks ", len(src_blocks))
        # gpu_cache_addr = [(kv_cache[0], kv_cache[1]) for kv_cache in self.gpu_cache_addr]
        
        with torch.cuda.stream(self.recv_streams[channel]):
            print("before recv blocks ", channel, opposite_rank)
            gpu_ops.RecvBlocks(self.gpu_cache_addr, src_blocks, self.cache_size_per_block, opposite_rank)
            print("after  recv blocks ", channel, opposite_rank)

            event = torch.cuda.Event()
            event.record()
        self.recv_events[channel] = (request_id, event)

    def send_blocks(self, channel: str, request_id: str, dst_blocks: List[int], opposite_rank: int) -> str: 
        if channel not in self.send_streams:
            self.send_streams[channel] = torch.cuda.Stream(device=torch.cuda.current_device())
        # print("send_blocks ", len(dst_blocks))
        # gpu_cache = [(kv_cache[0], kv_cache[1]) for kv_cache in self.gpu_cache]
        
        with torch.cuda.stream(self.send_streams[channel]):
            tensor_of_request_id = torch.Tensor([int(data, 16) for data in list(request_id)]).byte().cuda()
            self.send_waiting_request_ids[request_id] = tensor_of_request_id
            print("before send request ", channel, opposite_rank)
            gpu_ops.SendRequest(tensor_of_request_id.data_ptr(), self.request_id_size, opposite_rank)
            print("before send blocks ", channel, opposite_rank)
            gpu_ops.SendBlocks(self.gpu_cache_addr, dst_blocks, self.cache_size_per_block, opposite_rank)
            print("after send blocks ", channel, opposite_rank)
            event = torch.cuda.Event()
            event.record() 
        if channel not in self.send_events:
            self.send_events[channel] = [(request_id, event)]
        else:
            self.send_events[channel].append((request_id, event))

    #todo Tuple
    def check_send_finished_events(self) -> List[TransferTaskMeta]:
        #process send events
        send_finished_events: List[Tuple[str, List[int]]] = []
        send_blocks_finished: List[TransferTaskMeta] = []
        for channel, request_ids_and_events in self.send_events.items():
            num_finished_events = 0
            for request_id, event in request_ids_and_events:
                print("query request check_send_finished_events ", channel, request_id)
                if event.query():
                    send_blocks_finished.append(TransferTaskMeta(channel, request_id))
                    # 删除request tensor
                    del self.send_waiting_request_ids[request_id]
                    num_finished_events += 1
                else:
                    break
            send_finished_events.append((channel, num_finished_events))

        for channel, num_finished_events in send_finished_events:
            while num_finished_events != 0:
                print("finishd  check_send_finished_events ", channel)
                self.send_events[channel].pop(0)
                num_finished_events -= 1
        
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
                    print("finshed request id check_recv_finished_events ", channel, finished_request_id)
                else:
                    recv_blocks_finished.append(TransferTaskMeta(channel,request_id))
                    print("finshed blocks check_recv_finished_events ", channel)
                    # gpu_ops.HandleNcclCommDestroy()
        # release recv events
        for channel in recv_finished_events:
            self.recv_events.pop(channel)
        
        return recv_request_id_finished, recv_blocks_finished
    

def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
