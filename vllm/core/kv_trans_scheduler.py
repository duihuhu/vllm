from typing import Dict, List, Tuple
from enum import Enum
import threading

class TransferTaskMeta:
    def __init__(
        self,
        channel: str,
        request_id: str
    ) -> None:
        self.channel = channel
        self.request_id = request_id
        
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

class PrefillScheduleOutputs:
    def __init__(
        self,
        task_for_send_blocks: TransferBlocksTask
    ) -> None:
        self.task_for_send_blocks = task_for_send_blocks

class DecodeScheduleOutputs:
    def __init__(
        self,
        task_for_recv_request_id: TransferRequestIdTask,
        task_for_recv_blocks: TransferBlocksTask
    ) -> None:
        self.task_for_recv_request_id = task_for_recv_request_id
        self.task_for_recv_blocks = task_for_recv_blocks
        
class PrefillKvTransScheduler:
    """Prefill Manages the KV cache sending
    
    This class is responsible for manage and schedule send kv requests
    """
    def __init__(
        self,
        num_workers: int
    ) -> None:
        self.is_channel_sending: Dict[str, bool] = {}
        self.channel_requests_num: Dict[str, int] = {}
        self.channel_requests_ids: Dict[str, List[str]] = {}
        self.channel_ranks: Dict[str, List[int]] = {}
        
        self.finished_worker_count: Dict[str, int] = {}
        self.block_ids: Dict[str, List[int]] = {}
        self.num_workers = num_workers
    
    def add_kv_request(
        self,
        request_id: str,
        global_ranks: List[int],
        blocks: List[int]
    ) -> None:
        self.block_ids[request_id] = blocks
        self.finished_worker_count[request_id] = self.num_workers
        opp_rank_str = "_".join([str(rank) for rank in global_ranks])
        if opp_rank_str not in self.channel_ranks:
            self.channel_ranks[opp_rank_str] = global_ranks
            self.channel_requests_ids[opp_rank_str] = []
            self.channel_requests_num[opp_rank_str] = 0
            self.is_channel_sending[opp_rank_str] = True
            
        self.channel_requests_ids[opp_rank_str].append(request_id)
        self.channel_requests_num[opp_rank_str] += 1
    
    def _get_task_for_send_blocks(self) -> TransferBlocksTask:
        #调度最多requests的且可以使用的channel,并且FIFO调度请求
        sort_items = sorted(self.channel_requests_num.items(), key=lambda x:x[1], reverse=True)
        scheduled_channel  = None
        for sort_item in sort_items:
            channel = sort_item[0]
            if self.is_channel_sending[channel] and self.channel_requests_num[channel] != 0:
                scheduled_channel = channel
                break
        if not scheduled_channel:
            return None

        scheduled_ranks = self.channel_ranks[scheduled_channel]
        scheduled_request = self.channel_requests_ids[channel][0]
        blocks = self.block_ids[scheduled_request]
        self.is_channel_sending[scheduled_channel] = False
        return TransferBlocksTask(TransferTaskMeta(scheduled_channel,scheduled_request), scheduled_ranks, blocks)
        
    def schedule(self) -> PrefillScheduleOutputs:
        #一轮调度每种类型只生成一个请求
        task_for_send_blocks = self._get_task_for_send_blocks()
        return PrefillScheduleOutputs(task_for_send_blocks)
    
    def _process_send_blocks_finished(
        self,
        send_blocks_finished: List[TransferTaskMeta]
    ) -> List[str]:
        real_finished_req_ids = []
        for task_meta in send_blocks_finished:
            self.finished_worker_count[task_meta.request_id] -= 1
            if self.finished_worker_count[task_meta.request_id] == 0:
                self.channel_requests_ids[task_meta.channel].remove(task_meta.request_id)
                self.channel_requests_num[task_meta.channel] -= 1
                del self.block_ids[task_meta.request_id]
                del self.finished_worker_count[task_meta.request_id]
                self.is_channel_sending[task_meta.channel] = True
                real_finished_req_ids.append(task_meta.request_id)
        return real_finished_req_ids

    def add_finished_tasks(
        self,
        send_blocks_finished: List[TransferTaskMeta],
    ) -> List[str]:
        real_finished_req_ids = self._process_send_blocks_finished(send_blocks_finished)
        return real_finished_req_ids

class DecodeKvTransScheduler:
    """Decode Manages the KV cache recving
    
    This class is responsible for manage and schedue recv requests/
    """
    
    def __init__(
        self,
        num_workers: int
    ) -> None:
        self.is_channel_recving: Dict[str, bool] = {}
        self.channel_requests_num: Dict[str, int] = {}
        self.channel_requests_ids: Dict[str, List[str]] = {}
        self.channel_ranks: Dict[str, List[int]] = {}
        
        self.finished_worker_count: Dict[str, int] = {}
        self.block_ids: Dict[str, List[int]] = {}
        self.recv_waiting_requests: List[TransferTaskMeta] = []
        self.num_workers = num_workers
        
    def add_kv_request(
        self,
        request_id: str,
        opp_ranks: List[int],
        blocks: List[int]
    ) -> None:
        self.block_ids[request_id] = blocks
        self.finished_worker_count[request_id] = self.num_workers
        # print(" opp_ranks ", opp_ranks)
        opp_rank_str = "_".join([str(rank) for rank in opp_ranks])
        if opp_rank_str not in self.channel_ranks:
            self.channel_ranks[opp_rank_str] = opp_ranks
            self.channel_requests_ids[opp_rank_str] = []
            self.channel_requests_num[opp_rank_str] = 0
            self.is_channel_recving[opp_rank_str] = True
        self.channel_requests_ids[opp_rank_str].append(request_id)
        self.channel_requests_num[opp_rank_str] += 1
    
    def _get_task_for_recv_request_id(self) -> TransferRequestIdTask:
        #调度最多requests的且可以使用的channel
        sort_items = sorted(self.channel_requests_num.items(), key=lambda x:x[1], reverse=True)
        scheduled_channel = None
        for sort_item in sort_items:
            channel = sort_item[0]
            if self.is_channel_recving[channel] and self.channel_requests_num[channel] != 0:
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
                                  self.block_ids[task_meta.request_id])
    
    def schedule(self) -> DecodeScheduleOutputs:
        # 一轮调度每种类型只生成一个请求
        task_for_recv_request_id = self._get_task_for_recv_request_id()
        task_for_recv_blocks = self._get_task_for_recv_blocks()
        return DecodeScheduleOutputs(task_for_recv_request_id, task_for_recv_blocks)

    def _process_recv_request_id_finished(
        self,
        recv_request_id_finished: List[TransferTaskMeta]
    ) -> None:
        for task_meta in recv_request_id_finished:
            self.finished_worker_count[task_meta.request_id] -= 1
            if self.finished_worker_count[task_meta.request_id] == 0:
                self.recv_waiting_requests.append(task_meta)
                self.finished_worker_count[task_meta.request_id] = self.num_workers

    def _process_recv_blocks_finished(
        self,
        recv_blocks_finished: List[TransferTaskMeta]
    ) -> List[str]:
        real_finished_req_ids = []
        for task_meta in recv_blocks_finished:
            self.finished_worker_count[task_meta.request_id] -= 1
            if self.finished_worker_count[task_meta.request_id] == 0:
                self.channel_requests_ids[task_meta.channel].pop(0)
                self.channel_requests_num[task_meta.channel] -= 1
                del self.block_ids[task_meta.request_id]
                del self.finished_worker_count[task_meta.request_id]
                self.is_channel_recving[task_meta.channel] = True
                real_finished_req_ids.append(task_meta.request_id)
        return real_finished_req_ids

    def add_finished_tasks(
        self,
        recv_request_id_finished: List[TransferTaskMeta],
        recv_blocks_finished: List[TransferTaskMeta]
    ) -> List[str]:
        self._process_recv_request_id_finished(recv_request_id_finished)
        real_finished_req_ids = self._process_recv_blocks_finished(recv_blocks_finished)
        return real_finished_req_ids
