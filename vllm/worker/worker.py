"""A GPU worker class."""
import gc
import os
from typing import Dict, List, Tuple, Set, Optional

import torch
import torch.distributed


# from torch.cuda import (get_device_name, current_device, empty_cache, reset_peak_memory_stats, set_device,
#                         synchronize, max_memory_allocated, get_device_capability)

from vllm.sequence import SequenceData
from vllm.utils import get_max_shared_memory_bytes

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, LoRAConfig, DeployConfig)
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils.communication_op import (
    broadcast_tensor_dict)
from vllm.model_executor.parallel_utils.custom_all_reduce import init_custom_ar
from vllm.model_executor.parallel_utils.parallel_state import (
    ensure_model_parallel_initialized)
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner import ModelRunner
from vllm.lora.request import LoRARequest
from vllm._C import gpu_ops 
from vllm.logger import init_logger
import ray
import json
import socket
from vllm.core.kv_trans_scheduler import TransferTaskMeta, TransferRequestIdTask, TransferBlocksTask

logger = init_logger(__name__)

class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        deploy_config: DeployConfig = None,
        lora_config: Optional[LoRAConfig] = None,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        device_id: Optional[int] = 0,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.is_driver_worker = is_driver_worker
        self.deploy_config = deploy_config
        self.device_id = device_id
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        self.model_runner = ModelRunner(model_config,
                                        parallel_config,
                                        scheduler_config,
                                        lora_config=self.lora_config,
                                        kv_cache_dtype=kv_cache_dtype,
                                        is_driver_worker=is_driver_worker)
        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None

    def _get_local_device_info(self, rank_table_file: str):
        local_server_ip = socket.gethostbyname(socket.gethostname())
        local_rank = int(ray.get_runtime_context().get_accelerator_ids()["GPU"][0])
        with open(rank_table_file, 'r') as rank_table_reader:
            rank_table = json.load(rank_table_reader)
            for server_info in rank_table.get("server_list"):
                server_ip = server_info.get("server_id")
                if server_ip != local_server_ip:
                    continue
                for device_info in server_info.get("device"):
                    device_id = int(device_info.get("device_id"))
                    if device_id == local_rank:
                        global_rank = int(device_info.get("rank_id"))
                        return local_rank, global_rank

    def init_model(self) -> None:
        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        
        # if self.rank_table_file:
        #     self.local_rank, self.global_rank = self._get_local_device_info(self.deploy_config.rank_table_file)
        #     logger.info("rank = %d, local rank = %d global rank = %d", self.rank, self.local_rank, self.global_rank )
        # else:
        
        if not self.is_driver_worker:
            self.get_local_rank, self.global_rank = int(ray.get_runtime_context().get_accelerator_ids()["GPU"][0]), None
            logger.info("worker get from rank = %d, ", self.get_local_rank)
        else:
            self.get_local_rank = self.device_id

        # self.device = torch.device(f"cuda:{self.local_rank}")
        # torch.cuda.set_device(self.device)

        self.device = torch.device(f"cuda:{self.get_local_rank}")
        torch.cuda.set_device(self.device)
        
        _check_if_gpu_supports_dtype(self.model_config.dtype)
        
        logger.info("self.rank = %d , self.local_rank = %d ", self.rank, self.local_rank)
        # Initialize the distributed environment.
        init_distributed_environment(self.parallel_config, self.get_local_rank,
                                     self.distributed_init_method)
        
        if not self.parallel_config.disable_custom_all_reduce:
            init_custom_ar()
        # Initialize the model.
        set_random_seed(self.model_config.seed)
        
        #todo hucc CreateGlobalNcclComm 
        # if self.deploy_config.rank_table_file:
        #     if CreateGlobalNcclComm(self.deploy_config.rank_table_file, self.global_rank) !=0:
        #         raise ValueError("CreateHcclFromRankTable error")
        
        if gpu_ops.CreateGlobalNcclComm(self.get_local_rank, 4) !=0:
            print("self.local_rank ", self.get_local_rank)
            raise ValueError("CreateHcclFromRankTable error")

    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
        cache_dtype: str,
    ) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model and returns the maximum
        number of GPU and CPU cache blocks that can be allocated.

        Args:
            block_size: The size of the cache block.
            gpu_memory_utilization: The fraction of the total GPU memory to use.
            cpu_swap_space: The size of the CPU swap space in bytes.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        self.model_runner.profile_run()

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        peak_memory = total_gpu_memory - free_gpu_memory

        cache_block_size = CacheEngine.get_cache_block_size(
            block_size, cache_dtype, self.model_config, self.parallel_config)
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory) //
            cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()
        gc.collect()
        torch.cuda.empty_cache()
        return num_gpu_blocks, num_cpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache
        self.model_runner.set_block_size(self.cache_engine.block_size)

    def warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache)
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def cache_swap(
        self,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        cache_events = self.cache_events if issued_cache_op else None

        # Wait for cache operations to finish.
        # TODO(woosuk): Profile swapping overhead and optimize if needed.
        if cache_events is not None:
            for event in cache_events:
                event.wait()

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None,
        blocks_to_swap_in: Optional[Dict[int, int]] = None,
        blocks_to_swap_out: Optional[Dict[int, int]] = None,
        blocks_to_copy: Optional[Dict[int, List[int]]] = None,
        wait_for_swap_out: List[str] = None,
    ) -> Tuple[SamplerOutput, Tuple[List[str], List[str]]]:
        if self.is_driver_worker:
            assert seq_group_metadata_list is not None
            num_seq_groups = len(seq_group_metadata_list)
            assert blocks_to_swap_in is not None
            assert blocks_to_swap_out is not None
            assert blocks_to_copy is not None
            data = {
                "num_seq_groups": num_seq_groups,
                "blocks_to_swap_in": blocks_to_swap_in,
                "blocks_to_swap_out": blocks_to_swap_out,
                "blocks_to_copy": blocks_to_copy,
            }
            broadcast_tensor_dict(data, src=0)
        else:
            data = broadcast_tensor_dict(src=0)
            num_seq_groups = data["num_seq_groups"]
            blocks_to_swap_in = data["blocks_to_swap_in"]
            blocks_to_swap_out = data["blocks_to_swap_out"]
            blocks_to_copy = data["blocks_to_copy"]

        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)
        
        #todo hucc
        if wait_for_swap_out:
            self.cache_engine.wait_for_swap_out_events(wait_for_swap_out)
        
        if not seq_group_metadata_list:
            swap_finished_req_ids = self.cache_engine.check_finished_events()
    
            return ([[]], swap_finished_req_ids)
        #
        
        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return {}

        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.gpu_cache)
        
        swap_finished_req_ids = self.cache_engine.check_finished_events()
        
        return (output, swap_finished_req_ids)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    def decode_recv_request_id(
        self,
        task: TransferRequestIdTask
    ) -> str:
        self.cache_engine.recv_request_id(task.channel, task.opposite_ranks[self.rank])
    
    def prefill_send_blocks(
        self,
        task: TransferBlocksTask
    ) -> None:
        task_meta = task.meta
        self.cache_engine.send_blocks(task_meta.channel, task_meta.request_id,
                                      task.blocks, task.opposite_ranks[self.rank])
    
    def decode_recv_blocks(
        self,
        task: TransferBlocksTask
    ) -> None:
        task_meta = task.meta
        self.cache_engine.recv_blocks(task_meta.channel, task_meta.request_id,
                                      task.blocks, task.opposite_ranks[self.rank])
        
    def check_prefill_finished_transfer_task(self) -> Tuple[List[TransferTaskMeta], List[TransferTaskMeta]]:
        send_blocks_finished = self.cache_engine.check_send_finished_events()
        return send_blocks_finished
    
    def check_decode_finished_transfer_task(self) -> List[TransferTaskMeta]:
        recv_request_id_finished, recv_blocks_finished = self.cache_engine.check_recv_finished_events()
        return recv_request_id_finished, recv_blocks_finished

    #hucc
    # def remote_recv_request(
    #     self,
    #     channel: str,
    #     remote_ranks: List[int]
    # ) -> str:
    #     self.cache_engine.remote_recv_request(channel, remote_ranks[self.rank])
    
    # def remote_send_blocks(
    #     self,
    #     channel: str,
    #     request_id: str,
    #     remote_ranks: List[int],
    #     blocks: List[int]
    # ) -> None:
    #     self.cache_engine.remote_send_blocks(channel, request_id, blocks, remote_ranks[self.rank])
    
    # def remote_recv_blocks(
    #     self,
    #     channel: str,
    #     request_id: str,
    #     remote_ranks: List[int],
    #     blocks: List[int]
    # ) -> None:
    #     self.cache_engine.remote_recv_blocks(channel, request_id, blocks, remote_ranks[self.rank])
    
    # def check_remote_trans_finished(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    #     recv_request_finished, send_data_finished = self.cache_engine.check_remote_send_finished_events()
    #     recv_data_finished = self.cache_engine.check_remote_recv_finished_events()
    #     return recv_request_finished, send_data_finished, recv_data_finished
    
def init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        print("distributed_init_method 2 ", distributed_init_method)
        
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size}).")
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized")
    else:
        print("distributed_init_method 3 ", distributed_init_method)
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )
    print("distributed_init_method 4 ", distributed_init_method)

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    # Check if the GPU supports the dtype.
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")
