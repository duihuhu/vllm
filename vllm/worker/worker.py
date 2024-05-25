"""A GPU worker class."""
import gc
import os
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.distributed

from vllm.sequence import SequenceData
from vllm.utils import get_max_shared_memory_bytes

from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, VisionLanguageConfig, DeployConfig)
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils import pynccl_utils
from vllm.model_executor.parallel_utils.communication_op import (
    broadcast_tensor_dict)
from vllm.model_executor.parallel_utils.custom_all_reduce import init_custom_ar
from vllm.model_executor.parallel_utils.parallel_state import (
    ensure_model_parallel_initialized)
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner import ModelRunner

from vllm._C import gpu_ops 
from vllm.logger import init_logger
import ray
import json
import socket
#no TransferRequestIdTask, TransferBlocksTask
from vllm.core.kv_trans_scheduler import TransferTaskMeta,  TransferRequestIdTask, TransferBlocksTask, TransferTask
from vllm.worker.comm_engine import CommEngine
import time
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
        device_config: DeviceConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        deploy_config: DeployConfig = None,
        lora_config: Optional[LoRAConfig] = None,
        vision_language_config: Optional[VisionLanguageConfig] = None,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        device_id: Optional[int] = 0,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.is_driver_worker = is_driver_worker
        self.deploy_config = deploy_config
        self.device_id = device_id
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        self.vision_language_config = vision_language_config
        if self.vision_language_config:
            assert not self.lora_config, (
                "To be tested: vision language model with LoRA settings.")

        self.model_runner = ModelRunner(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            lora_config=self.lora_config,
            kv_cache_dtype=kv_cache_dtype,
            is_driver_worker=is_driver_worker,
            vision_language_config=vision_language_config)
        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.cache_engine = None
        self.gpu_cache = None

        self.trans_blocks_time = 0
    def init_device(self) -> None:
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.model_config.dtype)
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        
        if not self.is_driver_worker:
            self.get_local_rank, self.global_rank = int(ray.get_runtime_context().get_accelerator_ids()["GPU"][0]), None
            logger.info("worker get from rank = %d, ", self.get_local_rank)
        else:
            self.get_local_rank = self.device_id
            
        # Initialize the distributed environment.
        init_distributed_environment(self.parallel_config, self.rank,
                                     self.distributed_init_method,
                                     self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)
        
        if self.deploy_config.enable_separate:
            if gpu_ops.CreateGlobalNcclComm(self.get_local_rank, 4, 0) !=0:
                # print("self.local_rank ", self.get_local_rank)
                raise ValueError("CreateNcclFromRankTable error")
        return self.get_local_rank

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
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        peak_memory = self.init_gpu_memory - free_gpu_memory
        assert peak_memory > 0, (
            "Error in memory profiling. This happens when the GPU memory was "
            "not properly cleaned up before initializing the vLLM instance.")

        cache_block_size = self.get_cache_block_size_bytes(
            block_size, cache_dtype)
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
                                        self.parallel_config, self.deploy_config, self.rank)
        self.gpu_cache = self.cache_engine.gpu_cache
        self.model_runner.set_block_size(self.cache_engine.block_size)

    def init_comm_engine(self):
        self.common_engine = CommEngine(self.cache_config, self.model_config, self.parallel_config, self.deploy_config, self.gpu_cache)
        
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
        # TODO(woosuk): Profile swapping overhead and optimize if needed.
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None,
        blocks_to_swap_in: Optional[Dict[int, int]] = None,
        blocks_to_swap_out: Optional[Dict[int, int]] = None,
        blocks_to_copy: Optional[Dict[int, List[int]]] = None,
        blocks_to_send_remote: Optional[Dict[str, Tuple[int, List[int], List[int]]]] = None,
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
                "blocks_to_send_remote": blocks_to_send_remote,
            }
            broadcast_tensor_dict(data, src=0)
        else:
            data = broadcast_tensor_dict(src=0)
            num_seq_groups = data["num_seq_groups"]
            blocks_to_swap_in = data["blocks_to_swap_in"]
            blocks_to_swap_out = data["blocks_to_swap_out"]
            blocks_to_copy = data["blocks_to_copy"]
            blocks_to_send_remote = data["blocks_to_send_remote"]

        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        #todo hucc
        if wait_for_swap_out:
            self.cache_engine.wait_for_swap_out_events(wait_for_swap_out)
                
        if num_seq_groups == 0:
            swap_finished_req_ids = self.cache_engine.check_finished_events()
    
            return ([[]], swap_finished_req_ids)
        
        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return {}

        output = self.model_runner.execute_model(seq_group_metadata_list,
                                                 self.gpu_cache, blocks_to_send_remote, self.cache_engine)
        
        swap_finished_req_ids = self.cache_engine.check_finished_events()
        return (output, swap_finished_req_ids)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()
    
    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_cache_block_size_bytes(self, block_size: int,
                                   cache_dtype: str) -> int:
        """Get the size of the KV cache block size in bytes.
        """
        return CacheEngine.get_cache_block_size(block_size, cache_dtype,
                                                self.model_config,
                                                self.parallel_config)
    def trans_blocks(
        self,
        send_tasks: List[TransferTask],
        recv_tasks: List[TransferTask]
    ) -> None:
        # t1 = time.time()
        for task in send_tasks:
            task_meta = task.meta
            self.common_engine.send_blocks(task_meta.channel, task_meta.request_id, task.blocks, task.opposite_ranks[self.rank])
        for task in recv_tasks:
            task_meta = task.meta
            self.common_engine.recv_blocks(task_meta.channel, task_meta.request_id, task.blocks, task.opposite_ranks[self.rank])
        # t2 = time.time()
        # self.trans_blocks_time = self.trans_blocks_time + t2 - t1
        # print("trans_blocks time ", self.trans_blocks_time)

    
    def check_finished_transfer_task(self) -> List[TransferTaskMeta]:
        send_blocks_finished = self.common_engine.check_send_finished_events()
        recv_blocks_finished = self.common_engine.check_recv_finished_events()
        return send_blocks_finished, recv_blocks_finished
                
def init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
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
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    if pynccl_utils.is_initialized():
        pynccl_world_size = pynccl_utils.get_world_size()
        if pynccl_world_size != parallel_config.world_size:
            raise RuntimeError(
                "pynccl is already initialized but the pynccl world "
                "size does not match parallel_config.world_size "
                f"({pynccl_world_size} vs. {parallel_config.world_size}).")
    elif parallel_config.world_size > 1:
        # NOTE(woosuk): We don't initialize pynccl process group when world size
        # is 1.
        pynccl_utils.init_process_group(
            world_size=parallel_config.world_size,
            local_rank=local_rank,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    if pynccl_utils.is_initialized():
        pynccl_utils.all_reduce(torch.zeros(1).cuda())
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    # Initialize a custom fast all-reduce implementation.
    if not parallel_config.disable_custom_all_reduce:
        init_custom_ar()


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

