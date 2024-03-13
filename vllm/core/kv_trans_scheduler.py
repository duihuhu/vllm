from typing import Dict, List, Tuple
from enum import Enum
import threading

class SchedulerOutput:
    def __init__(
        self,
        channel_for_recv_request: Tuple[str, List[int]],
        request_for_send_data: Tuple[str, str, List[int], List[int]],
        request_for_recv_data: Tuple[str, str, List[int], List[int]]    
        ) -> None:
            self.channel_for_recv_request = channel_for_recv_request
            self.request_for_send_data = request_for_send_data
            self.request_for_recv_data = request_for_recv_data
            
    def empty(self) -> bool:
        return not self.channel_for_recv_request and not self.request_for_send_data \
            and not self.request_for_recv_data 

class KvTransScheduler:
    """Manage the KV cache Transfering
    
    This class is responsible for manage and schedule transfer kv requests.
    """

    def __init__(
        self,
        num_workers: int 
    ) -> None:
        self.channel_send_useful: Dict[str, bool] = {}
        self.channel_recv_useful: Dict[str, bool] = {}
        self.channel_requests_num: Dict[str, List[int]] = {}
        self.channel_requests: Dict[str, Tuple[List[str], List[str]]] = {}
        self.channel_ranks: Dict[str, List[int]] = {}

        self.request_finished_worker_count: Dict[str, int] = {}
        self.request_data: Dict[str, List[Tuple[List[int], List[int]]]] = {}
        self.request_for_send_data_ready: List[Tuple[str, str]] = []
        
        self.num_workers = num_workers
        self.scheduer_lock = threading.Lock()
        
    def register_kv_request(
        self,
        request_id: str,
        send_kv: bool,
        opp_ranks: List[int],
        blocks: List[int]
    ) -> None:
        with self.scheduer_lock:
            self.request_data[request_id] = blocks
            self.request_finished_worker_count[request_id] = self.num_workers
            opp_rank_str = "_".join([str[rank] for rank in opp_ranks])
            if opp_rank_str not in self.channel_ranks:
                self.channel_ranks[opp_rank_str] = opp_ranks
                self.channel_requests[opp_rank_str] = ([], [])
                self.channel_requests_num[opp_rank_str] = [0, 0]
                self.channel_send_useful[opp_rank_str] = True
                self.channel_recv_useful[opp_rank_str] = True
            if send_kv:
                self.channel_requests[opp_rank_str][0].append(request_id)
                self.channel_requests_num[opp_rank_str][0] += 1
            else:
                self.channel_requests[opp_rank_str][1].append(request_id)
                self.channel_requests_num[opp_rank_str][1] += 1
                
    def _get_channel_for_recv_request(self) -> Tuple(str, List[int]):
        #调度最多requests的且可以使用的channel
        sort_items = sorted(self.channel_requests_num.items(), key=lambda x:x[1][0], reverse=True)
        scheduled_channel = None
        for sort_item in sort_items:
            channel = sort_item[0]
            if self.channel_send_useful[channel] and self.channel_requests_num[channel][0] != 0:
                scheduled_channel = sort_item[0]
                break
        if not scheduled_channel:
            return None
        
        scheduled_ranks = self.channel_ranks[scheduled_channel]
        self.channel_send_useful[scheduled_channel] = False
        return scheduled_channel, scheduled_ranks
    
    def _get_request_for_send_data(self) -> Tuple(str, str, List[int], List[int]):
        #调度最早收到的request的请求，准备发数据
        if not self.request_for_send_data_ready:
            return None
        channel, request_id = self.request_for_send_data_ready.pop(0)
        return channel, request_id, self.channel_ranks[channel], self.request_data[request_id]
    
    def _get_request_for_recv_data(self) -> Tuple[str, str, List[int], List[int]]:
        #调度最多requests且可以使用channel,并且按照FIFO调度
        sort_items = sorted(self.channel_requests_num.items(), key=lambda x:x[1][1], reverse=True)
        scheduled_channel= None
        for sort_item in sort_items:
            channel = sort_item[0]
            if self.channel_recv_useful[channel] and self.channel_requests_num[channel][1] !=0:
                scheduled_channel = channel
                break
        if not scheduled_channel:
            return None
 
        scheduled_ranks = self.channel_ranks[scheduled_channel]
        scheduled_request = self.channel_requests[channel][1][0]
        blocks = self.request_data[scheduled_request]
        self.channel_recv_useful[scheduled_channel] = False
        return scheduled_channel, scheduled_request, scheduled_ranks, blocks
    
    def schedule(self) -> SchedulerOutput:
        #一轮调度美中类型只生成一个请求
        channel_for_recv_request = self._get_channel_for_recv_request()
        request_for_send_data = self._get_request_for_send_data()
        request_for_recv_data = self._get_request_for_recv_data()
        return SchedulerOutput(channel_for_recv_request, request_for_send_data, request_for_recv_data)
    
    def _process_recv_request_finished(
        self,
        recv_request_finished: List[Tuple[str, str]]
    ) -> List[str]:
        for (channel, request_id) in recv_request_finished:
            self.request_finished_worker_count[request_id] -= 1
            if self.request_finished_worker_count[request_id] == 0:
                self.request_for_send_data_ready.append((channel, request_id))
                self.request_finished_worker_count[request_id] = self.num_workers
                
    def _process_send_data_finished(
        self,
        send_data_finished: List[Tuple[str, str]]
    ) -> List[str]:
        finished_req_ids = []
        for (channel, request_id) in send_data_finished:
            self.request_finished_worker_count[request_id] -= 1
            finished_req_ids.append(request_id)
            if self.request_finished_worker_count[request_id] == 0:
                self.channel_requests[channel][0].remove(request_id)
                self.channel_requests_num[channel][0] -= 1
                del self.request_data[request_id]
                del self.request_finished_worker_count[request_id]
                self.channel_send_useful[channel] = True
        return finished_req_ids
    
    def _process_recv_data_finished(
        self,
        recv_data_finished: List[Tuple[str, str]]
    ) -> List[str]:
        finished_req_ids = []
        for (channel, request_id) in recv_data_finished:
            self.request_finished_worker_count[request_id] -= 1
            finished_req_ids.append(request_id)
        if self.request_finished_worker_count[request_id] == 0:
            self.channel_requests[channel][1].pop(0)
            self.channel_requests_num[channel][1] -= 1
            del self.request_data[request_id]
            del self.request_finished_worker_count[request_id]
            self.channel_recv_useful[channel] = True
        return finished_req_ids
            
    def register_finished_requests(
        self,
        finished_requests: Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]
    ) -> Tuple[List[str], List[str]]:
        self._process_recv_request_finished(finished_requests[0])
        send_finished_req_ids = self._process_send_data_finished(finished_requests[1])
        recv_finished_req_ids = self._process_recv_data_finished(finished_requests[2])
        return send_finished_req_ids, recv_finished_req_ids