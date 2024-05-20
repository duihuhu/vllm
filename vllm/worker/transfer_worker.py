
from multiprocessing import Process, Pipe
from vllm.worker.comm_engine import CommEngine
import enum
from vllm._C import gpu_ops 
import torch 
class TaskType(enum.Enum):
    # CREATE_NCCL = enum.auto()
    TRANSFER_SEND = enum.auto()
    TRANSFER_RECV_ID = enum.auto()
    TRANSFER_RECV_BLOCKS = enum.auto()
    TRANSFER_CHECK_FINISHED = enum.auto()

class TransferWorker:
    def __init__(self, gpu_cache_addr, cache_config, model_config, parallel_config, deploy_config, rank, local_rank, nccl_local_rank) -> None:

        self.gpu_cache_addr = gpu_cache_addr
        
        self.comm_engine = CommEngine(cache_config, model_config, parallel_config, deploy_config, gpu_cache_addr)

        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.deploy_config = deploy_config
        
        self.task_queue_parent, self.task_queue_child = Pipe()
        self.result_queue_parent, self.result_queue_child = Pipe()
        self.rank = rank
        self.local_rank = local_rank
        self.nccl_local_rank = nccl_local_rank
        
        self.execute = Process(target=self.worker)
        self.execute.start()

    def init_device(self) -> None:
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

    def worker(self):
        self.init_device()
        if gpu_ops.CreateGlobalNcclComm(self.nccl_local_rank, 4, 0) !=0:
            # print("self.local_rank ", self.get_local_rank)
            raise ValueError("CreateNcclFromRankTable error")
        while True:
            # 接收任务
            task_type, task = self.task_queue_child.recv()
            if task_type == TaskType.TRANSFER_SEND:
                task_meta = task.meta
                self.comm_engine.send_blocks(task_meta.channel, task_meta.request_id, task.blocks, task.opposite_ranks[self.rank])
            elif task_type == TaskType.TRANSFER_RECV_ID:
                self.comm_engine.recv_request_id(task.channel, task.opposite_ranks[self.rank])
            elif task_type == TaskType.TRANSFER_RECV_BLOCKS:
                task_meta = task.meta
                self.comm_engine.recv_blocks(task_meta.channel, task_meta.request_id, task.blocks, task.opposite_ranks[self.rank])
            elif task_type == TaskType.TRANSFER_CHECK_FINISHED:
                send_blocks_finished = self.comm_engine.check_send_finished_events()
                recv_request_id_finished, recv_blocks_finished = self.comm_engine.check_recv_finished_events()
                self.result_queue_child.send((send_blocks_finished, recv_request_id_finished, recv_blocks_finished))
            else:
                raise RuntimeError("invalid task_type.")

    def add_task(self, task):
        self.task_queue_parent.send(task)

    def get_batch_finished_task(self):
        finished_tasks = []
        while self.result_queue_parent.poll():
            finished_tasks.append(self.result_queue_parent.recv())
        return finished_tasks
    
    def get_finished_task(self):
        if self.result_queue_parent.poll():
            return self.result_queue_parent.recv()
        return None
