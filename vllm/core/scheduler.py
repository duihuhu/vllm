import enum
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig, DeployConfig
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.core.policy import PolicyFactory
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceStatus)

logger = init_logger(__name__)


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


# seq_group: SequenceGroup to schedule.
# token_chunk_size: The number of prefill tokens to be processed in the next
# step.
@dataclass
class ScheduledSequenceGroup:
    # A sequence group that's scheduled.
    seq_group: SequenceGroup
    # The total chunk size (number of tokens) to process for next iteration.
    # 1 for decoding. Same as prompt tokens for prefill, but if prefill is
    # chunked, it can be smaller than that.
    token_chunk_size: int


class SchedulerOutputs:

    def __init__(
        self,
        scheduled_seq_groups: Iterable[ScheduledSequenceGroup],
        prompt_run: bool,
        num_batched_tokens: int,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        ignored_seq_groups: List[SequenceGroup],
    ) -> None:
        """A list of sequence groups to be scheduled as a single batch.

        Args:
            scheduled_seq_groups: A tuple of scheduled sequence group and its
                token chunk size.
            prompt_run: True if all sequence groups are in prefill phase.
                If False, all sequence groups are in decoding phase.
            num_batched_tokens: Total number of batched tokens.
            blocks_to_swap_in: Blocks to swap in. Dict of CPU -> GPU block
                number.
            blocks_to_swap_out: Blocks to swap out. Dict of GPU -> CPU block
                number.
            blocks_to_copy: Blocks to copy. Source to a list of dest blocks.
            ignored_seq_groups: Sequence groups that are going to be ignored.
        """
        # A tuple of scheduled sequence group and its chunk size.
        self.scheduled_seq_groups: ScheduledSequenceGroup = scheduled_seq_groups
        # True if all sequence groups are in prefill phase. If False, all
        # sequence groups are in decoding phase.
        self.prompt_run: bool = prompt_run
        # Total number of batched tokens.
        self.num_batched_tokens: int = num_batched_tokens
        # Blocks to swap in. Dict of CPU -> GPU block number.
        self.blocks_to_swap_in: Dict[int, int] = blocks_to_swap_in
        # Blocks to swap out. Dict of GPU -> CPU block number.
        self.blocks_to_swap_out: Dict[int, int] = blocks_to_swap_out
        # Blocks to copy. Source to a list of dest blocks.
        self.blocks_to_copy: Dict[int, List[int]] = blocks_to_copy
        # Sequence groups that are going to be ignored.
        self.ignored_seq_groups: List[SequenceGroup] = ignored_seq_groups

        # Swap in and swap out should never happen at the same time.
        assert not (blocks_to_swap_in and blocks_to_swap_out)

        self.num_loras: int = len(self.lora_requests)
        if self.num_loras > 0:
            self._sort_by_lora_ids()

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)

    def _sort_by_lora_ids(self) -> bool:
        self.scheduled_seq_groups = sorted(
            self.scheduled_seq_groups,
            key=lambda g: (g.seq_group.lora_int_id, g.seq_group.request_id))

    @property
    def lora_requests(self) -> Set[LoRARequest]:
        return {g.seq_group.lora_request for g in self.scheduled_seq_groups}


class SwappingSequenceGroup:
    def __init__(
        self,
        seq_group: SequenceGroup,
        num_swapping_workers: int,
        ) -> None:
            self.seq_group = seq_group
            self.num_swapping_workers = num_swapping_workers

class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        deploy_config: DeployConfig, 
        lora_config: Optional[LoRAConfig],
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.deploy_config = deploy_config
        # Note for LoRA scheduling: the current policy is extremely
        # simple and NOT fair. It can lead to starvation of some
        # LoRAs. This should be improved in the future.
        self.lora_config = lora_config

        self.prompt_limit = min(self.scheduler_config.max_model_len,
                                self.scheduler_config.max_num_batched_tokens)

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")

        BlockSpaceManagerImpl = BlockSpaceManager.get_block_space_manager_class(
            version="v2" if self.scheduler_config.
            use_v2_block_manager else "v1")

        # Create the block space manager.
        self.block_manager = BlockSpaceManagerImpl(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching,
            enable_mcache=self.cache_config.enable_mcache,
            enable_radix_caching=self.cache_config.enable_radix_caching)

        # Sequence groups in the WAITING state.
        self.waiting: Deque[SequenceGroup] = deque()
        # Sequence groups in the RUNNING state.
        self.running: Deque[SequenceGroup] = deque()
        # Sequence groups in the SWAPPED state.
        self.swapped: Deque[SequenceGroup] = deque()

        self.decode: Deque[SequenceGroup] = deque()

        # Time at previous scheduling step
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        self.prev_prompt = False
        # Latency of the last prompt step
        self.last_prompt_latency = 0.0
        
        self.swap_finished_req_ids: List[Tuple[List[str], List[str]]] = []
        self.swapping_in: List[SwappingSequenceGroup] = []
        self.swapping_out: List[SwappingSequenceGroup] = []
        
        self.send_finished_req_ids: List[str] = []
        self.recv_finished_req_ids: List[str] = []
        
        self.send_transfering: Dict[str, SequenceGroup] = {}
        self.recv_transfering: Dict[str, SequenceGroup] = {}
        
        self.req_send_transfering: Dict[str, int] = {}
        
        self.swaping_req_id: List[str] = []
        self.num_workers: int = 0

    @property
    def lora_enabled(self) -> bool:
        return bool(self.lora_config)

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def add_send_finished(self, request_ids: List[str]):
        self.send_finished_req_ids.extend(request_ids)
    
    def add_send_transfering(self, seq_group: SequenceGroup) -> None:
        #Add sequence groups to the send transfering map.
        print("add_send_transfering ", seq_group.request_id, time.time())
        self.send_transfering[seq_group.request_id] = seq_group
    
    #todo check free_seq
    #del_send_transfering: 分配block失败的时候删除
    def del_send_transfering(self, request_id: str) -> None:
        # Delete sequence groups to the send  transfering map 
        if request_id in self.send_transfering:
            seq = self.send_transfering[request_id].get_seqs()[0]
            # self.free_seq(seq)
            del self.send_transfering[request_id]
    
    def get_send_transfering(self, request_id: str) -> None:
        if request_id not in self.send_transfering:
            return None
        return self.send_transfering[request_id]

    def add_recv_finished(self, request_ids: List[str]):
        print("recv_finished_req_ids ", request_ids, time.time())
        self.recv_finished_req_ids.extend(request_ids)
    
    def add_swap_finished(self, request_ids: List[str]):
        self.swap_finished_req_ids.extend(request_ids)

    def add_recv_transfering(self, seq_group: SequenceGroup) -> None:
        #Add sequence groups to the recv transfering map
        self.recv_transfering[seq_group.request_id] = seq_group

    def add_swap_in_req_id(self, request_id) -> None:
        self.swaping_req_id.append(request_id)

    def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a sequence group with the given ID.

        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
        """
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)
        for state_queue in [self.waiting, self.running, self.swapped]:
            aborted_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                if not request_ids:
                    # Using 'break' here may add two extra iterations,
                    # but is acceptable to reduce complexity .
                    break
                if seq_group.request_id in request_ids:
                    # Appending aborted group into pending list.
                    aborted_groups.append(seq_group)
                    request_ids.remove(seq_group.request_id)
            for aborted_group in aborted_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_group)
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)

    def has_unfinished_seqs(self) -> bool:
        return (self.waiting or self.running or self.swapped or
                self.swapping_in or self.swapping_out)

    def fetch_decoded_seq_groups(self) -> List[SequenceGroup]:
        decoded_seq_groups = []
        while self.decoded:
            decoded_seq_groups.append(self.decoded.pop())
        return decoded_seq_groups

    def check_hbm_usage(self) -> Tuple[str, int]:
        hbm_ratio = self.block_manager.gpu_allocator.get_num_free_blocks() / self.block_manager.num_total_gpu_blocks

        if hbm_ratio > self.block_manager.waterswap_blocks:
            num_blocks = int((hbm_ratio - self.block_manager.waterswap_blocks * 1.2) * self.block_manager.num_total_gpu_blocks)
            return num_blocks
        return 0


    def evict_hbm_caches(self, num_blocks):
        cache_blocks_to_swap_out: Dict[int, int] = {}
        mapping = self.block_manager.evict_hbm_caches(num_blocks)
        cache_blocks_to_swap_out.update(mapping)
        return cache_blocks_to_swap_out

    def fetch_prefilled_seq_groups(self) -> List[SequenceGroup]:
        prefilled_seq_groups = []
        while self.running:
            prefilled_seq_groups.append(self.running.popleft())
        return prefilled_seq_groups
    
    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)
    
    # def allocate_only_kv_blocks(self, seq_group: SequenceGroup) -> List[int]:
    #     seq = seq_group.get_seqs()[0]
    #     if not self.block_manager.can_allocate(seq_group):
    #         return None
    #     else:
    #         self._allocate_only_kv_blocks(seq_group)
    #         # self.block_manager.block_tables[seq.seq_id]
    #         block_table = self.block_manager.kv_block_tables[seq.seq_id]
    #         phy_blocks = [phy_block for phy_block in block_table]
    #         return phy_blocks
        
    def allocate_kv_blocks(self, seq_group: SequenceGroup) -> List[int]:
        seq = seq_group.get_seqs()[0]
        if not self.block_manager.can_allocate(seq_group):
            return None
        else:
            blocks_to_swap_in = self._allocate_kv_blocks(seq_group)
            # self.block_manager.block_tables[seq.seq_id]
            block_table = self.block_manager.kv_block_tables[seq.seq_id]

            phy_blocks = [phy_block for phy_block in block_table]
            return phy_blocks, blocks_to_swap_in
    
    def allocate_kv_radix_blocks(self, seq_group: SequenceGroup) -> List[int]:
        seq = seq_group.get_seqs()[0]
        if not self.block_manager.can_allocate(seq_group):
            return None
        else:
            blocks_to_swap_in = self._allocate_kv_radix_blocks(seq_group)
            block_table = self.block_manager.kv_block_tables[seq.seq_id]

            phy_blocks = [phy_block for phy_block in block_table]
            return phy_blocks, blocks_to_swap_in
    
    def fetch_kv_blocks(self, seq_group: SequenceGroup) -> List[int]:
        seq = seq_group.get_seqs()[0]
        block_table = self.block_manager.block_tables[seq.seq_id]
        blocks = [phy_block.block_number for phy_block in block_table]
        # for bkt in block_table:
            # print("fetch_kv_blocks ", bkt.device, bkt.computed)
        return blocks

    def _schedule(self) -> SchedulerOutputs:
        # Blocks that need to be swapped or copied before model execution.
        cached_seq_groups: List[SequenceGroup] = []
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        now = time.time()
        
        cache_blocks_to_swap_out = self._check_tranfer_finished_req()
        
        # Join waiting sequences if possible.
        if not self.swapped:
            ignored_seq_groups: List[SequenceGroup] = []
            scheduled: List[SequenceGroup] = []
            # The total number of sequences on the fly, including the
            # requests in the generation phase.
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)
            curr_loras = set(
                seq_group.lora_int_id
                for seq_group in self.running) if self.lora_enabled else None

            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            leftover_waiting_sequences = deque()
            num_batched_tokens = 0
            while self._passed_delay(now) and self.waiting:
                seq_group = self.waiting[0]
                print("seq_group request id prefill start time ", seq_group.request_id, time.time())
                waiting_seqs = seq_group.get_seqs(
                    status=SequenceStatus.WAITING)
                assert len(waiting_seqs) == 1, (
                    "Waiting sequence group should have only one prompt "
                    "sequence.")
                # get_len includes output tokens if the request has been
                # preempted.
                num_prefill_tokens = waiting_seqs[0].get_len()
                if num_prefill_tokens > self.prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prefill_tokens} tokens) is too "
                        f"long and exceeds limit of {self.prompt_limit}")
                    for seq in waiting_seqs:
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.popleft()
                    continue
                if self.deploy_config.enable_cache_meta:
                    if not seq_group.cache_meta or not seq_group.cache_meta.ready:
                        # If the sequence group cannot be allocated, stop.
                        can_allocate = self.block_manager.can_allocate(seq_group)
                        if can_allocate == AllocStatus.LATER:
                            break
                        elif can_allocate == AllocStatus.NEVER:
                            logger.warning(
                                f"Input prompt ({num_prefill_tokens} tokens) is too "
                                f"long and exceeds the capacity of block_manager")
                            for seq in waiting_seqs:
                                seq.status = SequenceStatus.FINISHED_IGNORED
                            ignored_seq_groups.append(seq_group)
                            self.waiting.popleft()
                            continue
                        
                lora_int_id = 0
                if self.lora_enabled:
                    lora_int_id = seq_group.lora_int_id
                    if (lora_int_id > 0 and lora_int_id not in curr_loras
                            and len(curr_loras) >= self.lora_config.max_loras):
                        # We don't have a space for another LoRA, so
                        # we ignore this request for now.
                        leftover_waiting_sequences.appendleft(seq_group)
                        self.waiting.popleft()
                        continue
                
                if self.deploy_config.enable_cache_meta:
                    if seq_group.cache_meta and not seq_group.cache_meta.ready:
                        self._allocate(seq_group, blocks_to_swap_in)
                        seq = seq_group.get_seqs()[0]
                        block_table = self.block_manager.block_tables[seq.seq_id]
                        phy_blocks = [phy_block for phy_block in block_table]
                        computed_blocks = [phy_block.block_number for phy_block in phy_blocks if phy_block.computed == True]
                        if len(computed_blocks) < seq_group.cache_meta.cmeta_kv_len:
                            seq_group.cache_meta.cached_len = len(computed_blocks)
                            cached_seq_groups.append(seq_group)
                            self.waiting.popleft()
                            continue    

                # If the number of batched tokens exceeds the limit, stop.
                num_batched_tokens += num_prefill_tokens
                if (num_batched_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                if lora_int_id > 0:
                    curr_loras.add(lora_int_id)
                self.waiting.popleft()
                # self._allocate(seq_group)
                if not self.deploy_config.enable_cache_meta:
                    self._allocate(seq_group, blocks_to_swap_in)
                elif self.deploy_config.enable_cache_meta:
                    if seq_group.cache_meta and seq_group.cache_meta.ready:
                        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
                            seq.status = SequenceStatus.RUNNING
                    
                # print("_allocate_mixed_cache blocks_to_swap_in ", blocks_to_swap_in)
                self.running.append(seq_group)
                num_curr_seqs += num_new_seqs
                scheduled.append(
                    ScheduledSequenceGroup(
                        seq_group=seq_group,
                        token_chunk_size=num_prefill_tokens))
            self.waiting.extendleft(leftover_waiting_sequences)
            # print("waiting seq  blocks_to_swap_in ", blocks_to_swap_in)
            if scheduled or ignored_seq_groups:
                self.prev_prompt = True
                scheduler_outputs = SchedulerOutputs(
                    scheduled_seq_groups=scheduled,
                    prompt_run=True,
                    num_batched_tokens=num_batched_tokens,
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=ignored_seq_groups,
                )
                return scheduler_outputs, cache_blocks_to_swap_out, cached_seq_groups

        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        self.running = self.policy.sort_by_priority(now, self.running)

        # Reserve new token slots for the running sequence groups.
        running: Deque[SequenceGroup] = deque()
        preempted: List[SequenceGroup] = []
        while self.running:
            seq_group = self.running.popleft()
            while not self.block_manager.can_append_slot(seq_group):
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = self.running.pop()
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq_group, blocks_to_copy)
                running.append(seq_group)
        self.running = running

        # Swap in the sequence groups in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        if not preempted:
            num_curr_seqs = sum(seq_group.get_max_num_running_seqs()
                                for seq_group in self.running)
            curr_loras = set(
                seq_group.lora_int_id
                for seq_group in self.running) if self.lora_enabled else None

            leftover_swapped = deque()

            while self.swapped:
                seq_group = self.swapped[0]
                lora_int_id = 0
                if self.lora_enabled:
                    lora_int_id = seq_group.lora_int_id
                    if (lora_int_id > 0 and lora_int_id not in curr_loras
                            and len(curr_loras) >= self.lora_config.max_loras):
                        # We don't have a space for another LoRA, so
                        # we ignore this request for now.
                        leftover_swapped.appendleft(seq_group)
                        self.swapped.popleft()
                        continue

                # If the sequence group cannot be swapped in, stop.
                if not self.block_manager.can_swap_in(seq_group):
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.get_max_num_running_seqs()
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                if lora_int_id > 0:
                    curr_loras.add(lora_int_id)
                self.swapped.popleft()
                self._swap_in(seq_group, blocks_to_swap_in)
                self._append_slot(seq_group, blocks_to_copy)
                num_curr_seqs += num_new_seqs
                self.running.append(seq_group)

            self.swapped.extendleft(leftover_swapped)

        # Each sequence in the generation phase only takes one token slot.
        # Therefore, the number of batched tokens is equal to the number of
        # sequences in the RUNNING state.
        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running)

        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=[
                ScheduledSequenceGroup(seq_group=running_group,
                                       token_chunk_size=1)
                for running_group in self.running
            ],
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],
        )
        return scheduler_outputs, cache_blocks_to_swap_out, cached_seq_groups

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, List[SequenceGroup]]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_outputs, cache_blocks_to_swap_out, cached_seq_groups = self._schedule()
        now = time.time()

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)

            # seq_id -> SequenceData
            seq_data: Dict[int, SequenceData] = {}
            # seq_id -> physical block numbers
            block_tables: Dict[int, List[int]] = {}
            seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
            for seq in seqs:
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                if not self.block_manager.enable_radix_caching:
                    block_tables[seq_id] = self.block_manager.get_block_table(seq)
                    self.block_manager.access_all_blocks_in_seq(seq, now)
                else:
                    block_table = self.block_manager.block_tables[seq.seq_id]
                    block_tables[seq_id] = seq.computed_block + \
                        [block.block_number for block in block_table[len(seq.computed_block):]] 

            if self.block_manager.enable_radix_caching:
                common_computed_block_nums = (
                    self.block_manager.get_common_computed_block_ids_one_seq(seqs[0]))
            else:
                common_computed_block_nums = (
                    self.block_manager.get_common_computed_block_ids(
                        seq_group.get_seqs(status=SequenceStatus.RUNNING)))

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=scheduler_outputs.prompt_run,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
                token_chunk_size=token_chunk_size,
                lora_request=seq_group.lora_request,
                computed_block_nums=common_computed_block_nums,
                state=seq_group.state,
                # `multi_modal_data` will only be present for the 1st comm
                # between engine and worker.
                # the subsequent comms can still use delta, but
                # `multi_modal_data` will be None.
                multi_modal_data=seq_group.multi_modal_data
                if scheduler_outputs.prompt_run else None,
            )
            seq_group_metadata_list.append(seq_group_metadata)

        # Now that the batch has been created, we can assume all blocks in the
        # batch will have been computed before the next scheduling invocation.
        # This is because the engine assumes that a failure in model execution
        # will crash the vLLM instance / will not retry.
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group)
        return seq_group_metadata_list, scheduler_outputs, cache_blocks_to_swap_out, cached_seq_groups

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        """Free a sequence from a block table."""
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        self.decoded = deque(seq_group for seq_group in self.running
                             if seq_group.is_finished())
        
        self.running = deque(seq_group for seq_group in self.running
                             if not seq_group.is_finished())

    # alread merge into _allocate_kv_blocks
    # def _allocate_only_kv_blocks(self, seq_group: SequenceGroup) -> None:
    #     self.block_manager.allocate_only_kv_blocks(seq_group)
        
    def _allocate_kv_blocks(self, seq_group: SequenceGroup) -> None:
        blocks_to_swap_in = {}
        if self.deploy_config.role == "prompt":
            self.block_manager.allocate_only_kv_blocks(seq_group)
        elif self.deploy_config.role == "decoder":
            blocks_to_swap_in = self.block_manager.allocate_kv_blocks(seq_group)
            for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
                seq.status = SequenceStatus.RUNNING
            return blocks_to_swap_in
    
    def _allocate_kv_radix_blocks(self, seq_group: SequenceGroup) -> None:
        blocks_to_swap_in = self.block_manager.allocate_radix_cache(seq_group, True)
        if self.deploy_config.role == "decoder":
            for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
                seq.status = SequenceStatus.RUNNING
        return blocks_to_swap_in
    
    # alread merge into _allocate
    # def _allocate_mixed_cache(self, seq_group: SequenceGroup,  blocks_to_swap_in: Dict[int, int] = {}) -> None:
    #     mapping = self.block_manager.allocate_mixed_cache(seq_group)
    #     blocks_to_swap_in.update(mapping)
    #     for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
    #         seq.status = SequenceStatus.RUNNING
            
    def _allocate(self, seq_group: SequenceGroup,  blocks_to_swap_in: Dict[int, int] = {}) -> None:
        if self.block_manager.enable_radix_caching:
            self.block_manager.allocate_radix_cache(seq_group)
        elif self.block_manager.enable_mcache:
            mapping = self.block_manager.allocate_mixed_cache(seq_group)
            blocks_to_swap_in.update(mapping)
        else:
            self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _append_slot(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            ret = self.block_manager.append_slot(seq)
            if ret is not None:
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> None:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not currently supported. In
        # such a case, we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.free_seq(seq)
            seq.reset_state_for_recompute()
        # NOTE: For FCFS, we insert the preempted sequence group to the front
        # of the waiting queue.
        self.waiting.appendleft(seq_group)

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped.append(seq_group)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: Dict[int, int],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED

    def _passed_delay(self, now: float) -> bool:
        if self.prev_prompt:
            self.last_prompt_latency = now - self.prev_time
        self.prev_time, self.prev_prompt = now, False
        # Delay scheduling prompts to let waiting queue fill up
        if self.scheduler_config.delay_factor > 0 and self.waiting:
            earliest_arrival_time = min(
                [e.metrics.arrival_time for e in self.waiting])
            passed_delay = (
                (now - earliest_arrival_time) >
                (self.scheduler_config.delay_factor * self.last_prompt_latency)
                or not self.running)
        else:
            passed_delay = True
        return passed_delay

    #kv缓存传输完了
    def _check_tranfer_finished_req(self) -> None:
        for request_id in self.send_finished_req_ids[:]:
            if request_id in self.req_send_transfering:
                del self.req_send_transfering[request_id]
                blocks = self.block_manager.req_block_tables[request_id]
                for block in blocks:
                    block.ref_count = block.ref_count - 1
                del self.block_manager.req_block_tables[request_id]
                self.send_finished_req_ids.remove(request_id)
                continue
            
            seq_group = self.send_transfering[request_id]
            seq = seq_group.get_seqs()[0]
            # self.free_seq(seq)

            #should free 
            block_table = self.block_manager.block_tables[seq.seq_id]
            if self.block_manager.enable_radix_caching:
                for bkt in block_table:
                    self.block_manager.gpu_allocator.free_radix_cache(bkt)
            else:
                for bkt in block_table:
                    self.block_manager.gpu_allocator.free(bkt)
            del self.block_manager.block_tables[seq.seq_id]
            
            if request_id in self.send_transfering:
                del self.send_transfering[request_id]
                
            self.send_finished_req_ids.remove(request_id)
        
            # print("after send gpu can evicted blocks ", self.block_manager.gpu_allocator.get_num_can_evicted_blocks())

            # num_blocks = self.block_manager.gpu_allocator.get_num_can_evicted_blocks()
            # if num_blocks:
            #     cache_blocks_to_swap_out = self.evict_hbm_caches(num_blocks)
            #     print("cache_blocks_to_swap_out ", cache_blocks_to_swap_out)


        for request_id in self.recv_finished_req_ids[:]:
            if request_id in self.swaping_req_id and request_id not in self.swap_finished_req_ids:
                continue
            seq_group = self.recv_transfering[request_id]

            if self.deploy_config.role == "decoder":
                print("in decoder running append ", time.time())
                self.running.append(seq_group)
                #end recv, when role is decode
                #move kv_block_tables to block_table
                #move cached_kv_blocks to cached_blocks
                self.block_manager.move_kv_blocks_meta(seq_group)
                self.block_manager.mark_blocks_as_computed(seq_group=seq_group)
            
            if self.deploy_config.enable_cache_meta:
                if seq_group.cache_meta:
                    seq_group.cache_meta.ready = True
                    for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                        seq.status = SequenceStatus.WAITING
                    self.waiting.append(seq_group)
                    del self.recv_transfering[request_id]
                    self.recv_finished_req_ids.remove(request_id)
                    self.block_manager.mark_trans_blocks_as_computed(seq_group=seq_group)
                    continue
            
            # recv cache in prompt is only cache, not has reference, so it will in evicted cache
            if self.deploy_config.role == "prompt":
                self.block_manager.move_kv_blocks_meta(seq_group)
                self.block_manager.mark_blocks_as_computed(seq_group=seq_group)
                for seq in seq_group.get_seqs():
                    self.block_manager.free(seq)                    
            del self.recv_transfering[request_id]
            self.recv_finished_req_ids.remove(request_id)
            if request_id in self.swaping_req_id:
                self.swaping_req_id.remove(request_id)
            if request_id in self.swap_finished_req_ids:
                self.swap_finished_req_ids.remove(request_id)
                
            # print("after recv gpu can evicted blocks ", self.block_manager.gpu_allocator.get_num_can_evicted_blocks())
            # #swap where
            if self.deploy_config.role == "prompt" and self.deploy_config.enable_mcache:
                num_blocks = self.block_manager.gpu_allocator.get_num_can_evicted_blocks()
                if num_blocks:
                    cache_blocks_to_swap_out = self.evict_hbm_caches(num_blocks)
                    print("prompt kv cache_blocks_to_swap_out ", cache_blocks_to_swap_out)
                    return cache_blocks_to_swap_out
        return None
