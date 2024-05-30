from typing import Dict, List, Tuple
import enum
import threading
import heapq
from vllm._C import trans_ops

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

# class TransferTask:
#     def __init__(
#         self,
#         meta: TransferTaskMeta,
#         opposite_ranks: List[int],
#         blocks: List[int],
#         type: TaskType
#     ):
#         self.meta = meta
#         self.opposite_ranks = opposite_ranks
#         self.blocks = blocks
#         self.type = type
        
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
        
class KvTransScheduler:
    """Prefill Manages the KV cache sending
    
    This class is responsible for manage and schedule send kv requests
    """
    def __init__(
        self,
        num_workers: int,
        enable_layer: bool = False
    ) -> None:
        self.is_channel_sending: Dict[str, bool] = {}
        self.channel_requests_num: Dict[str, List[int]] = {}
        self.channel_requests_ids: Dict[str, Tuple[List[str], List[str]]] = {}
        self.channel_ranks: Dict[str, List[int]] = {}
        
        self.send_finished_worker_count: Dict[str, int] = {}
        self.recv_finished_worker_count: Dict[str, int] = {}
        self.send_block_ids: Dict[str, List[int]] = {}
        self.recv_block_ids: Dict[str, List[int]] = {}
        self.num_workers = num_workers
        
        ############
        self.is_channel_recving: Dict[str, bool] = {}
        self.recv_waiting_requests: List[TransferTaskMeta] = []
        self.enable_layer = enable_layer

    def add_kv_request(
        self,
        request_id: str,
        global_ranks: List[int],
        blocks: List[int],
        is_send: bool,
    ) -> None:
        if is_send:
            self.send_block_ids[request_id] = blocks
            self.send_finished_worker_count[request_id] = self.num_workers
        else:
            self.recv_block_ids[request_id] = blocks
            self.recv_finished_worker_count[request_id] = self.num_workers
                      
        opp_rank_str = "_".join([str(rank) for rank in global_ranks])
        if opp_rank_str not in self.channel_ranks:
            self.channel_ranks[opp_rank_str] = global_ranks
            self.channel_requests_ids[opp_rank_str] = ([],[])
            self.channel_requests_num[opp_rank_str] = [0,0]
            self.is_channel_sending[opp_rank_str] = True
            self.is_channel_recving[opp_rank_str] = True
            
        if is_send:
            self.channel_requests_ids[opp_rank_str][0].append(request_id)
            self.channel_requests_num[opp_rank_str][0] += 1
        else:
            self.channel_requests_ids[opp_rank_str][1].append(request_id)
            self.channel_requests_num[opp_rank_str][1] += 1
            
    def _get_task_for_send_blocks(self) -> TransferBlocksTask:
        #调度最多requests的且可以使用的channel,并且FIFO调度请求
        sort_items = sorted(self.channel_requests_num.items(), key=lambda x:x[1][0], reverse=True)
        scheduled_channel  = None
        for sort_item in sort_items:
            channel = sort_item[0]
            if self.is_channel_sending[channel] and self.channel_requests_num[channel][0] != 0:
                scheduled_channel = channel
                break
        if not scheduled_channel:
            return None

        scheduled_ranks = self.channel_ranks[scheduled_channel]
        scheduled_request = self.channel_requests_ids[channel][0][0]
        blocks = self.send_block_ids[scheduled_request]
        self.is_channel_sending[scheduled_channel] = False
        return TransferBlocksTask(TransferTaskMeta(scheduled_channel, scheduled_request), scheduled_ranks, blocks)
    
    def _get_task_for_recv_request_id(self) -> TransferRequestIdTask:
        #调度最多requests的且可以使用的channel
        sort_items = sorted(self.channel_requests_num.items(), key=lambda x:x[1][1], reverse=True)
        scheduled_channel = None
        for sort_item in sort_items:
            channel = sort_item[0]
            if self.is_channel_recving[channel] and self.channel_requests_num[channel][1] != 0:
                scheduled_channel = sort_item[0]
                break
        if not scheduled_channel:
            return None
        
        scheduled_ranks = self.channel_ranks[scheduled_channel]
        self.is_channel_recving[scheduled_channel] = False
        return TransferRequestIdTask(scheduled_channel, scheduled_ranks)

    def _get_task_for_recv_blocks(self) -> TransferBlocksTask:
        #调度最早收到request的请求, 准备发数据
        if not self.recv_waiting_requests:
            return None
        task_meta = self.recv_waiting_requests.pop(0)
        return TransferBlocksTask(task_meta, self.channel_ranks[task_meta.channel],
                                  self.recv_block_ids[task_meta.request_id])
       
    def schedule(self) -> ScheduleOutputs:
        #一轮调度每种类型只生成一个请求
        task_for_send_blocks = self._get_task_for_send_blocks()
        task_for_recv_request_id = self._get_task_for_recv_request_id()
        task_for_recv_blocks = self._get_task_for_recv_blocks()
        return ScheduleOutputs(task_for_send_blocks, task_for_recv_request_id, task_for_recv_blocks)
    
    def _process_send_blocks_finished(
        self,
        send_blocks_finished: List[TransferTaskMeta]
    ) -> List[str]:
        real_finished_req_ids = []
        for task_meta in send_blocks_finished:
            if not self.enable_layer:
                self.send_finished_worker_count[task_meta.request_id] -= 1
                if self.send_finished_worker_count[task_meta.request_id] == 0:
                    # print("send request blocks finished ")
                    self.channel_requests_ids[task_meta.channel][0].remove(task_meta.request_id)
                    self.channel_requests_num[task_meta.channel][0] -= 1
                    del self.send_block_ids[task_meta.request_id]
                    del self.send_finished_worker_count[task_meta.request_id]
                    self.is_channel_sending[task_meta.channel] = True
                    real_finished_req_ids.append(task_meta.request_id)
            else:
                real_finished_req_ids.append(task_meta.request_id)
        return real_finished_req_ids

    def _process_recv_request_id_finished(
        self,
        recv_request_id_finished: List[TransferTaskMeta]
    ) -> None:
        for task_meta in recv_request_id_finished:
            self.recv_finished_worker_count[task_meta.request_id] -= 1
            if self.recv_finished_worker_count[task_meta.request_id] == 0:
                # print("recv request id finished ")
                self.recv_waiting_requests.append(task_meta)
                self.recv_finished_worker_count[task_meta.request_id] = self.num_workers

    def _process_recv_blocks_finished(
        self,
        recv_blocks_finished: List[TransferTaskMeta]
    ) -> List[str]:
        real_finished_req_ids = []
        for task_meta in recv_blocks_finished:
            self.recv_finished_worker_count[task_meta.request_id] -= 1
            if self.recv_finished_worker_count[task_meta.request_id] == 0:
                # print("recv request blocks finished ")
                self.channel_requests_ids[task_meta.channel][1].pop(0)
                self.channel_requests_num[task_meta.channel][1] -= 1
                del self.recv_block_ids[task_meta.request_id]
                del self.recv_finished_worker_count[task_meta.request_id]
                self.is_channel_recving[task_meta.channel] = True
                real_finished_req_ids.append(task_meta.request_id)
        return real_finished_req_ids

    def add_finished_tasks(
        self,
        send_blocks_finished: List[TransferTaskMeta],
        recv_blocks_finished: List[TransferTaskMeta]
    ) -> Tuple[List[str], List[str]]:
        real_send_finished_req_ids = self._process_send_blocks_finished(send_blocks_finished)
        real_recv_finished_req_ids = self._process_recv_blocks_finished(recv_blocks_finished)
        return real_send_finished_req_ids, real_recv_finished_req_ids
    
class SendKvTransferScheduler:
    def __init__(self,
                 num_workers,
                 enable_layer) -> None:
        self.channel_request_ids: Dict[str, List[PriorityRequest]] = {}
        
        self.finished_worker_count: Dict[str, int]  = {}
        self.block_ids: Dict[str, List[int]] = {}
        self.num_workers = num_workers
        
        self.opposite_ranks = list(range(num_workers, num_workers * 2))
        
        self.channel_transfer_tag: Dict[str, int] = {}
        
        self.enable_layer = enable_layer
    
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
                    meta=trans_ops.TransferTaskMeta(channel, request_id)
                    print("send meta ", type(meta), type(self.opposite_ranks), type(self.block_ids[request_id]))
                    tasks = trans_ops.TransferTask(
                        meta=meta,
                        opposite_ranks=self.opposite_ranks,
                        blocks=self.block_ids[request_id],
                        type=trans_ops.TaskType.TRANSFER_SEND_BLOCKS
                    )
                    print("send tasks ")
                    scheduled_transfer_tasks.append(trans_ops.TransferTask(
                        meta=trans_ops.TransferTaskMeta(channel, request_id),
                        opposite_ranks=self.opposite_ranks,
                        blocks=self.block_ids[request_id],
                        type=trans_ops.TaskType.TRANSFER_SEND_BLOCKS
                    ))
                    self.channel_transfer_tag[channel] += 1
                else:
                    break
        
        return scheduled_transfer_tasks
    
    def schedule(self) -> List[trans_ops.TransferTask]:
        return self._get_task_for_send_blocks()
    
    def _process_send_blocks_finished(
        self,
        send_finished_taks: List[TransferTaskMeta]
    ) -> List[str]:
        real_finished_req_ids = []
        for task_meta in send_finished_taks:
            self.finished_worker_count[task_meta.request_id] -=1
            if self.finished_worker_count[task_meta.request_id] == 0:
                del self.block_ids[task_meta.request_id]
                del self.finished_worker_count[task_meta.request_id]
                real_finished_req_ids.append(task_meta.request_id)
                
        return real_finished_req_ids
    
    def add_finished_tasks(
        self,
        send_finished_tasks: List[TransferTaskMeta],
    ) -> List[str]:
        return self._process_send_blocks_finished(send_finished_tasks)
    
class RecvKvTransScheduler:
    def __init__(self,
                num_workers,
                enable_layer) -> None:
        self.channel_request_ids: Dict[str, List[str]] = {}
        
        self.finished_worker_count: Dict[str, int]  = {}
        self.block_ids: Dict[str, List[int]] = {}
        self.num_workers = num_workers
        
        self.opposite_ranks = list(range(0, num_workers * 2))
        
        self.channel_transfer_tag: Dict[str, int] = {}
        self.enable_layer = enable_layer
        
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
                meta=trans_ops.TransferTaskMeta(channel, request_id)
                opposite_ranks = [0, 1]
                blocks = [3785, 3784, 3783, 3782]
                task = trans_ops.TransferTask(
                    meta=meta,
                    opposite_ranks=opposite_ranks,
                    blocks=blocks,
                    type=trans_ops.TaskType.TRANSFER_RECV_BLOCKS
                )
                print("recv task ")
                scheduled_transfer_tasks.append(trans_ops.TransferTask(
                    meta=trans_ops.TransferTaskMeta(channel, request_id),
                    opposite_ranks=self.opposite_ranks,
                    blocks=self.block_ids[request_id],
                    type=trans_ops.TaskType.TRANSFER_RECV_BLOCKS
                ))
        return scheduled_transfer_tasks 

    def schedule(self) -> List[trans_ops.TransferTask]:
        return self._get_task_for_recv_blocks()
    

    def _process_recv_blocks_finished(
        self,
        recv_finished_taks: List[TransferTaskMeta]
    ) -> List[str]:
        real_finished_req_ids = []
        for task_meta in recv_finished_taks:
            self.finished_worker_count[task_meta.request_id] -=1
            if self.finished_worker_count[task_meta.request_id] == 0:
                del self.block_ids[task_meta.request_id]
                del self.finished_worker_count[task_meta.request_id]
                real_finished_req_ids.append(task_meta.request_id)
                
        return real_finished_req_ids
    
    def add_finished_tasks(
        self,
        recv_finished_tasks: List[TransferTaskMeta],
    ) -> List[str]:
        return self._process_recv_blocks_finished(recv_finished_tasks)