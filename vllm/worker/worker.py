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
from vllm.outputs import MergeReqInfo
import numpy as np
from multiprocessing import shared_memory

from vllm._C import trans_ops, swap_ops, ops
from vllm.logger import init_logger
import ray
#no TransferRequestIdTask, TransferBlocksTask
from vllm.core.kv_trans_scheduler import TransferTaskMeta
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
        self.use_agg_block = self.deploy_config.use_agg_block
        self.block_size2 = self.deploy_config.block_size
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
            vision_language_config=vision_language_config,
            use_agg_block=self.use_agg_block,
            block_size=self.block_size2)
        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.cache_engine = None
        self.gpu_cache = None
        self.cpu_cache = None
        self.dst_cpu_cache = {}

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
            self.nccl_local_rank, self.global_rank = int(ray.get_runtime_context().get_accelerator_ids()["GPU"][0]), None
            logger.info("worker get from rank = %d, ", self.nccl_local_rank)
        else:
            self.nccl_local_rank = self.device_id
            
        # Initialize the distributed environment.
        init_distributed_environment(self.parallel_config, self.rank,
                                     self.distributed_init_method,
                                     self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)
        
        return self.nccl_local_rank

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
        self.cpu_cache = self.cache_engine.cpu_cache
        self.model_runner.set_block_size(self.cache_engine.block_size)
        if self.use_agg_block:
            self.caches_addresses_tensors_gpu = self.cache_engine.get_tensor_for_caches_address(gpu=True)
            self.caches_addresses_tensors_cpu = self.cache_engine.get_tensor_for_caches_address(gpu=False)
            self.gpu_blocks_address = self.cache_engine.get_blocks_address(gpu=True)
            self.cpu_blocks_address = self.cache_engine.get_blocks_address(gpu=False)

        else:
            self.caches_addresses_tensors_gpu = None
            self.caches_addresses_tensors_cpu = None

    def init_swap_manager(self):
        gpu_cache = [(kv_cache[0], kv_cache[1]) for kv_cache in self.gpu_cache]
        cpu_cache = [(kv_cache[0], kv_cache[1]) for kv_cache in self.cpu_cache]
        if not self.use_agg_block:
            self.swap_manager = swap_ops.SwapManager(self.cache_engine.cache_size_per_block, gpu_cache, cpu_cache, False, self.model_config.get_num_layers(self.parallel_config))
        else:
            self.swap_manager = swap_ops.SwapManager(self.cache_engine.cache_block_size, self.gpu_blocks_address, self.cpu_blocks_address)

    def init_trans_manager(self):
        if not self.use_agg_block:
            gpu_cache = [(kv_cache[0], kv_cache[1]) for kv_cache in self.gpu_cache]
            self.trans_manager = trans_ops.TransManager(self.cache_engine.cache_size_per_block, gpu_cache, self.rank, self.local_rank, self.nccl_local_rank, self.parallel_config.tensor_parallel_size, self.model_config.get_num_layers(self.parallel_config), self.cache_engine.cache_block_size, [])
        else:
            null_gpu_cache = [(torch.empty(1), torch.empty(1))]
            self.trans_manager = trans_ops.TransManager(self.cache_engine.cache_size_per_block, null_gpu_cache, self.rank, self.local_rank, self.nccl_local_rank, self.parallel_config.tensor_parallel_size, self.model_config.get_num_layers(self.parallel_config), self.cache_engine.cache_block_size, self.gpu_blocks_address)
        
    
    def warm_up_model(self) -> None:
        if not self.model_config.enforce_eager:
            self.model_runner.capture_model(self.gpu_cache, self.caches_addresses_tensors_gpu)
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
            if self.use_agg_block:
                self.cache_engine.swap_by_agg2_in(blocks_to_swap_in)
            else:
                self.cache_engine.swap_in(blocks_to_swap_in)
        if blocks_to_swap_out:
            if self.use_agg_block:
                self.cache_engine.swap_by_agg2_out(blocks_to_swap_out)
            else:
                self.cache_engine.swap_out(blocks_to_swap_out)
        if blocks_to_copy:
            if self.use_agg_block:
                self.cache_engine.copy_agg(self.caches_addresses_tensors_gpu,
                                           blocks_to_copy)
            else:
                self.cache_engine.copy(blocks_to_copy)
             
    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None,
        blocks_to_swap_in: Optional[Dict[int, int]] = None,
        blocks_to_swap_out: Optional[Dict[int, int]] = None,
        blocks_to_copy: Optional[Dict[int, List[int]]] = None,
        merge_reqs_info: Optional[List[MergeReqInfo]] = None,
        evicted_blocks_to_swap_out: Optional[Dict[int, int]] = None,
        swap_id:  Optional[int] = None,
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
                "merge_reqs_info": merge_reqs_info,
                "evicted_blocks_to_swap_out": evicted_blocks_to_swap_out,
                "swap_id": swap_id,
            }
            broadcast_tensor_dict(data, src=0)
        else:
            data = broadcast_tensor_dict(src=0)
            num_seq_groups = data["num_seq_groups"]
            blocks_to_swap_in = data["blocks_to_swap_in"]
            blocks_to_swap_out = data["blocks_to_swap_out"]
            blocks_to_copy = data["blocks_to_copy"]
            merge_reqs_info = data["merge_reqs_info"]
            evicted_blocks_to_swap_out = data["evicted_blocks_to_swap_out"]
            swap_id = data["swap_id"]

        self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        #todo hucc
        if evicted_blocks_to_swap_out:
            if self.use_agg_block:
                self.swap_manager.add_swap_tasks(
                swap_ops.SwapTask(swap_id, evicted_blocks_to_swap_out, swap_ops.SwapType.SWAP_OUT_FULL_BLOCKS))
            else:
                self.swap_manager.add_swap_tasks(
                    swap_ops.SwapTask(swap_id, evicted_blocks_to_swap_out, swap_ops.SwapType.SWAP_OUT_BLOCKS))
            
        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return {}
        if self.deploy_config.enable_layer:
            output = self.model_runner.execute_model(seq_group_metadata_list,
                                                    self.gpu_cache, 
                                                    self.caches_addresses_tensors_gpu,
                                                    merge_reqs_info, 
                                                    self.trans_manager)
        else:
            output = self.model_runner.execute_model(seq_group_metadata_list, self.gpu_cache, self.caches_addresses_tensors_gpu)
        #TODO change return res
        # swap_finished_req_ids = self.cache_engine.check_finished_events()
        return (output, [])

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

    def evict_blocks(self,
                     swap_id: str,
                     evicted_blocks_to_swap_out: Dict[int,int]) -> None:
        if self.use_agg_block:
            self.swap_manager.add_swap_tasks(swap_ops.SwapTask(swap_id, evicted_blocks_to_swap_out, swap_ops.SwapType.SWAP_OUT_FULL_BLOCKS))
        else:
            self.swap_manager.add_swap_tasks(swap_ops.SwapTask(swap_id, evicted_blocks_to_swap_out, swap_ops.SwapType.SWAP_OUT_BLOCKS))

    def trans_blocks(
        self,
        send_tasks: List[trans_ops.TransferTask],
        recv_tasks: List[trans_ops.TransferTask],
        swap_to_remote_tasks: List[trans_ops.TransferTask],
    ) -> None:
        if self.deploy_config.enable_debug:
            t1 = time.time()
        if send_tasks:
            self.trans_manager.add_tasks(send_tasks)
        if recv_tasks:
            for task in recv_tasks:
                tsk = trans_ops.TransferTask.deserialize(task)
                blocks = tsk.blocks
                for block in block:
                    print("block " , self.gpu_cache[block][-1])
            self.trans_manager.add_tasks(recv_tasks)   
        if swap_to_remote_tasks:
            self.trans_manager.add_tasks(swap_to_remote_tasks)
            
        if self.deploy_config.enable_debug:
            t2 = time.time()
            self.trans_blocks_time = self.trans_blocks_time + t2 - t1

    def get_nccl_id(
        self,
        dst_channel, worker_type)->None:
        nccl_id = self.trans_manager.get_nccl_id(dst_channel, worker_type)
        return nccl_id
    
    def create_comm(
        self,
        nccl_id,
        dst_channel,
        worker_type
    ) -> None:
        self.trans_manager.create_comm(nccl_id, dst_channel, worker_type)
        if self.deploy_config.enable_trans_to_dram:
            torch.cuda.empty_cache()
            if dst_channel not in self.dst_cpu_cache:
                self.get_dst_shm_rank(dst_channel)
                dst_tensor = self.restore_other_shared_cpu_cache(dst_channel)
                dst_cpu_cache = [(kv_cache[0], kv_cache[1]) for kv_cache in dst_tensor]
                self.dst_cpu_cache[dst_channel] = dst_cpu_cache
                dst_blocks_cpu_cache = []
                if not self.use_agg_block:
                    self.trans_manager.init_dst_cpu_cache(dst_channel, dst_cpu_cache, dst_blocks_cpu_cache)
                else:
                    null_dst_cpu_cache = [(torch.empty(1), torch.empty(1))]
                    blocks_address = [] 
                    for cache_block in dst_tensor:
                        blocks_address.append(cache_block.data_ptr())
                    self.trans_manager.init_dst_cpu_cache(dst_channel, null_dst_cpu_cache, blocks_address)
        
    def get_dst_shm_rank(self, dst_channel):
        # 将字符串分割成整数列表
        dst_ranks = [int(token) for token in dst_channel.split('_')]
        self.dst_shm_rank = dst_ranks[self.local_rank]
        return self.dst_shm_rank
    
    def get_trans_blocks_time(
        self,
    ) -> None:
        return self.trans_blocks_time
    
    def get_finished_transfer_tasks(self) -> List[List[Tuple[List[trans_ops.TransferTaskMeta],List[trans_ops.TransferTaskMeta]]]]:
        return self.trans_manager.get_finished_transfer_tasks() 
    
    def get_finished_swap_tasks(self) -> List[List[str]]:
        return self.swap_manager.get_finished_swap_tasks()
    
    
    def share_cpu_cache(self, global_ranks):
        self.deploy_config.set_global_ranks(global_ranks) 
        channel = "_".join([str(rank) for rank in self.deploy_config.global_ranks])
        # 将 Tensor 列表转换为 numpy 数组并计算每个 Tensor 的大小
        np_arrays = [tensor.cpu().numpy() for tensor in self.cpu_cache]
        self.tensor_sizes = [np_array.nbytes for np_array in np_arrays]

        # 计算总共需要的字节数
        total_bytes = sum(self.tensor_sizes)
        
        share_cpu_cache_name = channel + "_" + str(self.nccl_local_rank)
        # 创建共享内存
        self.shm = shared_memory.SharedMemory(name=share_cpu_cache_name, create=True, size=total_bytes)
        self.shm_name = self.shm.name

        # 将所有 Tensor 数据拷贝到共享内存中
        offset = 0
        for np_array in np_arrays:
            np_array_flat = np_array.flatten()
            np.copyto(np.ndarray(np_array_flat.shape, dtype=np_array_flat.dtype, buffer=self.shm.buf, offset=offset), np_array_flat)
            offset += np_array.nbytes
    
    def calculate_tensor_sizes(self):
        # 创建一个空的 Tensor 列表
        tensors = self.cache_engine._allocate_kv_cache(self.cache_engine.num_cpu_blocks, "cpu", self.use_agg_block)
        # 将 Tensor 列表转换为 numpy 数组并计算每个 Tensor 的大小
        np_arrays = [tensor.numpy() for tensor in tensors]
        tensor_sizes = [np_array.nbytes for np_array in np_arrays]
        return tensor_sizes
    
    def restore_other_shared_cpu_cache(self, dst_channel):
        tensor_sizes = self.calculate_tensor_sizes()
        dst_tensors = []
        index = 0
        shm = shared_memory.SharedMemory(name=dst_channel + "_" +str(self.dst_shm_rank))
        print("restore_other_shared_cpu_cache shm.size ", shm.size)        
        if self.deploy_config.use_agg_block:
            kv_cache_shape = self.cache_engine.attn_backend.get_kv_cache_shape(
                self.cache_engine.num_cpu_blocks, self.cache_engine.block_size, self.cache_engine.num_heads, self.cache_engine.head_size, self.cache_engine.num_layers)
        else:
            kv_cache_shape = self.cache_engine.attn_backend.get_kv_cache_shape(
                self.cache_engine.num_cpu_blocks, self.cache_engine.block_size, self.cache_engine.num_heads, self.cache_engine.head_size, None)
        
        shm_np_array = np.ndarray((self.shm.size,), dtype=np.uint8, buffer=self.shm.buf)
        for tensor_size in tensor_sizes:
            # 从共享内存中读取数据并恢复成 Torch Tensor
            tensor_flat_np_array = shm_np_array[index:index + tensor_size].view(np.uint8)
            tensor_np_array = np.ndarray(kv_cache_shape, dtype=np.float16, buffer=tensor_flat_np_array)
            # tensor = torch.from_numpy(tensor_np_array).to(self.device)
            tensor = torch.from_numpy(tensor_np_array)
            dst_tensors.append(tensor)
            index += tensor_size

        return dst_tensors
    
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

