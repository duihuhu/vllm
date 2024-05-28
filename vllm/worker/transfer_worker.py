
from multiprocessing import Process, Pipe
from vllm.worker.comm_engine import CommEngine
import enum
from vllm._C import gpu_ops 
import torch 
from queue import Queue
from threading import Thread
class TaskType(enum.Enum):
    # CREATE_NCCL = enum.auto()
    TRANSFER_SEND = enum.auto()
    TRANSFER_RECV_BLOCKS = enum.auto()
    TRANSFER_CHECK_FINISHED = enum.auto()
    TRANSFER_CHECK_SEND_FINISHED = enum.auto()
    TRANSFER_CHECK_RECV_FINISHED = enum.auto()

class TransferWorker:
    def __init__(self, gpu_cache, cache_config, model_config, parallel_config, deploy_config, rank, local_rank, nccl_local_rank) -> None:

        self.gpu_cache = gpu_cache
        
        self.comm_engine = CommEngine(cache_config, model_config, parallel_config, deploy_config, gpu_cache)

        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.deploy_config = deploy_config
        
        self.task_queue = Queue()
        self.transfer_result_queue = Queue()

        self.rank = rank
        self.local_rank = local_rank
        self.nccl_local_rank = nccl_local_rank
        
        self.execute = Thread(target=self.worker)
        self.execute.start()

    def init_device(self) -> None:
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

    def worker(self):
        self.init_device()
        if gpu_ops.CreateGlobalNcclComm(self.nccl_local_rank, 2, 0) !=0:
            # print("self.local_rank ", self.get_local_rank)
            raise ValueError("CreateNcclFromRankTable error")
        while True:
            # 接收任务
            for task_type, task in self.task_queue.get():              
                if task_type == TaskType.TRANSFER_SEND:
                    task_meta = task.meta
                    self.comm_engine.send_blocks(task_meta.channel, task_meta.request_id, task.blocks, task.opposite_ranks[self.rank])
                elif task_type == TaskType.TRANSFER_RECV_BLOCKS:
                    task_meta = task.meta
                    self.comm_engine.recv_blocks(task_meta.channel, task_meta.request_id, task.blocks, task.opposite_ranks[self.rank])
                elif task_type == TaskType.TRANSFER_CHECK_SEND_FINISHED:
                    send_blocks_finished = self.comm_engine.check_send_finished_events()
                    self.transfer_result_queue.put(send_blocks_finished)
                elif task_type == TaskType.TRANSFER_CHECK_FINISHED:
                    recv_blocks_finished = self.comm_engine.check_recv_finished_events()
                    self.transfer_result_queue.put(recv_blocks_finished)
                else:
                    raise RuntimeError("invalid task_type.")
                
    def add_tasks(self, tasks):
        self.task_queue.put(tasks)
    
    def get_transfer_results(self):
        finished_task = self.transfer_result_queue.get()
        return finished_task