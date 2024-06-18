from typing import Dict, List, Tuple
import enum
import heapq
from vllm._C import trans_ops
from vllm.sequence import SequenceGroup

class TaskType(enum.Enum):
    TRANSFER_SEND_BLOCKS = enum.auto()
    TRANSFER_RECV_BLOCKS = enum.auto()
    
class TransferTaskMeta:
    def __init__(
        self,
        channel: str,
        request_id: str
    ) -> None:
        self.channel = channel
        self.request_id = request_id

class TransferTask:
    def __init__(
        self,
        meta: TransferTaskMeta,
        opposite_ranks: List[int],
        blocks: List[int],
        type: TaskType
    ):
        self.meta = meta
        self.opposite_ranks = opposite_ranks
        self.blocks = blocks
        self.type = type
        
PriorityRequest = Tuple[int, int ]       
 
class TransferRequestIdTask:
    def __init__(
        self,
        channel:str,
        opposite_ranks: List[int]
    ) -> None:
        self.channel = channel
        self.opposite_ranks = opposite_ranks

class TransferBlocksTask:
    def __init__(
        self,
        meta: TransferTaskMeta,
        opposite_ranks: List[int],
        blocks: List[int]
    ) -> None:
        self.meta = meta
        self.opposite_ranks = opposite_ranks
        self.blocks = blocks

class ScheduleOutputs:
    def __init__(
        self,
        task_for_send_blocks: TransferBlocksTask,
        task_for_recv_request_id: TransferRequestIdTask,
        task_for_recv_blocks: TransferBlocksTask
    ) -> None:
        self.task_for_send_blocks = task_for_send_blocks
        self.task_for_recv_request_id = task_for_recv_request_id
        self.task_for_recv_blocks = task_for_recv_blocks
        
class SendKvTransferScheduler:
    def __init__(self,
                 num_workers,
                 enable_layer,
                 role,
                 use_agg_block) -> None:
        self.channel_request_ids: Dict[str, List[PriorityRequest]] = {}
        
        self.finished_worker_count: Dict[str, int]  = {}
        self.block_ids: Dict[str, List[int]] = {}
        self.num_workers = num_workers
        
        self.role = role
        # self.opposite_ranks = list(range(num_workers, num_workers * 2))
        
        self.channel_transfer_tag: Dict[str, int] = {}
        
        self.enable_layer = enable_layer
        self.use_agg_block = use_agg_block
        
        #to record send hbm and remote dram blocks ids 
        self.swap_block_ids: Dict[str, Tuple[List[int], List[int]]] = {}
        self.swap_channel_request_ids: Dict[str, List[str]] = {}

    def add_layer_kv_request(
        self,
        request_id: str,
        global_ranks: List[int],
        blocks: List[int],
    ) -> None:
        channel = "_".join([str(rank) for rank in global_ranks])
        self.block_ids[request_id] = blocks
        self.finished_worker_count[request_id] = self.num_workers
        #this may be not need
        if channel not in self.channel_request_ids:
            self.channel_request_ids[channel] = []
            self.channel_transfer_tag[channel] = 0

    def add_dram_kv_request(self,
        request_id: str,
        global_ranks: List[int],
        blocks: List[int],
        dst_blocks: List[int]) -> None:
        channel = "_".join([str(rank) for rank in global_ranks])
        self.swap_block_ids[request_id] = (blocks, dst_blocks)
        self.finished_worker_count[request_id] = self.num_workers
        if channel not in self.swap_channel_request_ids:
            self.swap_channel_request_ids[channel] = []
        self.swap_channel_request_ids[channel].append(request_id)
 
    def add_kv_request(
        self,
        request_id: str,
        global_ranks: List[int],
        blocks: List[int],
        transfer_tag: int
    ) -> None:
        channel = "_".join([str(rank) for rank in global_ranks])
        self.block_ids[request_id] = blocks
        self.finished_worker_count[request_id] = self.num_workers
        if channel not in self.channel_request_ids:
            self.channel_request_ids[channel] = []
            self.channel_transfer_tag[channel] = 0
        heapq.heappush(self.channel_request_ids[channel], (transfer_tag, request_id))
    

    def _get_task_for_send_blocks(self) -> List[trans_ops.TransferTask]:
        scheduled_transfer_tasks: List[trans_ops.TransferTask] = []
        for channel, priority_request in self.channel_request_ids.items():
            while priority_request:
                head_req_tag = priority_request[0][0]
                if head_req_tag == self.channel_transfer_tag[channel]:
                    request: PriorityRequest = heapq.heappop(priority_request)
                    request_id = request[1]
                    if self.use_agg_block:
                        scheduled_transfer_tasks.append(trans_ops.TransferTask(trans_ops.TransferTaskMeta(channel, request_id),self.block_ids[request_id], trans_ops.TaskType.TRANSFER_SEND_FULL_BLOCKS).serialize())
                    else:
                        scheduled_transfer_tasks.append(trans_ops.TransferTask(trans_ops.TransferTaskMeta(channel, request_id),self.block_ids[request_id], trans_ops.TaskType.TRANSFER_SEND_BLOCKS).serialize())
                    self.channel_transfer_tag[channel] += 1
                else:
                    break
        
        return scheduled_transfer_tasks
    
    def _get_task_for_swap_blocks(self) -> List[trans_ops.TransferTask]:
        scheduled_transfer_tasks: List[trans_ops.TransferTask] = []
        for channel, request_ids in self.swap_channel_request_ids.items():
            while request_ids:
                request_id = request_ids.pop(0)
                src_blocks = self.swap_block_ids[request_id][0]
                dst_blocks = self.swap_block_ids[request_id][1]
                if self.use_agg_block:
                    print("_get_task_for_swap_blocks full block ")
                    scheduled_transfer_tasks.append(trans_ops.TransferTask(trans_ops.TransferTaskMeta(channel, request_id),src_blocks, dst_blocks, trans_ops.TaskType.TRANSFER_HBM_TO_DRAM_FULL_BLOCKS).serialize())
                else:
                    scheduled_transfer_tasks.append(trans_ops.TransferTask(trans_ops.TransferTaskMeta(channel, request_id),src_blocks, dst_blocks, trans_ops.TaskType.TRANSFER_HBM_TO_DRAM_BLOCKS).serialize())        
        return scheduled_transfer_tasks
    
    def schedule(self) -> List[trans_ops.TransferTask]:
        return self._get_task_for_send_blocks()
    
    def schedule_swap_to_remote(self) -> List[trans_ops.TransferTask]:
        return self._get_task_for_swap_blocks() 
    
    def _process_send_blocks_finished(
        self,
        send_finished_taks: List[trans_ops.TransferTaskMeta]
    ) -> List[str]:
        real_finished_req_ids = []
        for task_meta in send_finished_taks:
            self.finished_worker_count[task_meta.request_id] -=1
            if self.finished_worker_count[task_meta.request_id] == 0:
                del self.block_ids[task_meta.request_id]
                del self.finished_worker_count[task_meta.request_id]
                real_finished_req_ids.append(task_meta.request_id)
                
        return real_finished_req_ids
    
    def _process_swap_remote_blocks_finished(
        self,
        swap_remote_finished_taks: List[trans_ops.TransferTaskMeta]
    ) -> List[str]:
        real_finished_swap_req_ids = []
        for task_meta in swap_remote_finished_taks:
            self.finished_worker_count[task_meta.request_id] -=1
            if self.finished_worker_count[task_meta.request_id] == 0:
                del self.swap_block_ids[task_meta.request_id]
                del self.finished_worker_count[task_meta.request_id]
                real_finished_swap_req_ids.append(task_meta.request_id)
                
        return real_finished_swap_req_ids
    
    def add_finished_swap_remote_tasks(self,
        swap_remote_finished_taks: List[trans_ops.TransferTaskMeta]):
        return self._process_swap_remote_blocks_finished(swap_remote_finished_taks)
    
    def add_finished_tasks(
        self,
        send_finished_tasks: List[trans_ops.TransferTaskMeta],
    ) -> List[str]:
        return self._process_send_blocks_finished(send_finished_tasks)
    
class RecvKvTransScheduler:
    def __init__(self,
                num_workers,
                enable_layer,
                role,
                use_agg_block,) -> None:
        self.channel_request_ids: Dict[str, List[str]] = {}
        
        self.finished_worker_count: Dict[str, int]  = {}
        self.block_ids: Dict[str, List[int]] = {}
        self.num_workers = num_workers
        
        # self.opposite_ranks = list(range(0, num_workers * 2))
        
        self.channel_transfer_tag: Dict[str, int] = {}
        self.enable_layer = enable_layer
        self.role = role

        if enable_layer:
            self.merage_reqs: Dict[str, List[SequenceGroup]] = {}
        
        self.use_agg_block = use_agg_block
        
    def add_layer_kv_request(
        self,
        request_id: str,
        global_ranks: List[int],
        blocks: List[List[int]],
    ) -> None:
        channel = "_".join([str(rank) for rank in global_ranks])
        self.block_ids[request_id] = blocks
        self.finished_worker_count[request_id] = self.num_workers
        if channel not in self.channel_request_ids:
            self.channel_request_ids[channel] = []
            self.channel_transfer_tag[channel] = 0
        self.channel_request_ids[channel].append(request_id)
        current_transfer_tag = self.channel_transfer_tag[channel]
        self.channel_transfer_tag[channel] += 1
        return current_transfer_tag
        
    def add_kv_request(
        self,
        request_id: str,
        global_ranks: List[int],
        blocks: List[int],
    ) -> None:
        channel = "_".join([str(rank) for rank in global_ranks])
        self.block_ids[request_id] = blocks
        self.finished_worker_count[request_id] = self.num_workers
        if channel not in self.channel_request_ids:
            self.channel_request_ids[channel] = []
            self.channel_transfer_tag[channel] = 0
        self.channel_request_ids[channel].append(request_id)
        current_transfer_tag = self.channel_transfer_tag[channel]
        self.channel_transfer_tag[channel] += 1
        return current_transfer_tag
    
    def _get_task_for_recv_blocks(self) -> List[trans_ops.TransferTask]:
        scheduled_transfer_tasks: List[trans_ops.TransferTask] = []
        for channel, request_ids in self.channel_request_ids.items():
            while request_ids:
                request_id = request_ids.pop(0)
                if self.enable_layer and self.role=="decoder":
                    blocks = []
                    for block_id in self.block_ids[request_id]:
                        blocks.extend(block_id)
                    scheduled_transfer_tasks.append(trans_ops.TransferTask(trans_ops.TransferTaskMeta(channel, request_id), blocks, trans_ops.TaskType.TRANSFER_RECV_LAYER_BLOCKS).serialize())
                else:
                    if self.use_agg_block:
                        scheduled_transfer_tasks.append(trans_ops.TransferTask(trans_ops.TransferTaskMeta(channel, request_id), self.block_ids[request_id], trans_ops.TaskType.TRANSFER_RECV_FULL_BLOCKS).serialize())
                    else:
                        scheduled_transfer_tasks.append(trans_ops.TransferTask(trans_ops.TransferTaskMeta(channel, request_id), self.block_ids[request_id], trans_ops.TaskType.TRANSFER_RECV_BLOCKS).serialize())
        return scheduled_transfer_tasks 

    def schedule(self) -> List[trans_ops.TransferTask]:
        return self._get_task_for_recv_blocks()
    
    def _process_recv_blocks_finished(
        self,
        recv_finished_tasks: List[trans_ops.TransferTaskMeta]
    ) -> List[str]:
        real_finished_req_ids = []
        for task_meta in recv_finished_tasks:
            self.finished_worker_count[task_meta.request_id] -=1
            if self.finished_worker_count[task_meta.request_id] == 0:
                del self.block_ids[task_meta.request_id]
                del self.finished_worker_count[task_meta.request_id]
                real_finished_req_ids.append(task_meta.request_id)
                
        return real_finished_req_ids
    
    def add_finished_tasks(
        self,
        recv_finished_tasks: List[trans_ops.TransferTaskMeta],
    ) -> List[str]:
        return self._process_recv_blocks_finished(recv_finished_tasks)
    
class RadixSwapScheduler:
    def __init__(self,
                num_workers) -> None:        
        self.finished_worker_count: Dict[str, int]  = {}
        self.num_workers = num_workers
        
    def add_swap_task(
        self,
        swap_id: str,
    ) -> None:
        print("add swap id ", swap_id)
        self.finished_worker_count[swap_id] = self.num_workers
    
    def _process_swap_blocks_finished(
        self,
        swap_finished_tasks: List[str]
    ) -> List[str]:
        real_finished_swap_ids = []
        for swap_id in swap_finished_tasks:
            self.finished_worker_count[swap_id] -=1
            if self.finished_worker_count[swap_id] == 0:
                del self.finished_worker_count[swap_id]
                real_finished_swap_ids.append(swap_id)   
        return real_finished_swap_ids
    
    def add_finished_tasks(
        self,
        swap_finished_tasks: List[str],
    ) -> List[str]:
        return self._process_swap_blocks_finished(swap_finished_tasks)
    