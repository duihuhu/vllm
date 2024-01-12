import enum
import time
from typing import Dict, List, Optional, Tuple
import math

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.block_manager import BlockSpaceManager
from vllm.core.policy import PolicyFactory
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceOutputs,
                           SequenceStatus)

logger = init_logger(__name__)

_LOGGING_INTERVAL_SEC = 5


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


class SchedulerOutputs:

    def __init__(
        self,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        # Swap in and swap out should never happen at the same time.
        assert not (blocks_to_swap_in and blocks_to_swap_out)

    def is_empty(self) -> bool:
        return (not self.blocks_to_swap_in and not self.blocks_to_swap_out
                and not self.blocks_to_copy)


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        log_stats: bool,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.log_stats = log_stats

        # Instantiate the scheduling policy.
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        self.block_manager = BlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
        )

        # Sequence groups in the WAITING state.
        self.waiting: List[SequenceGroup] = []
        # Sequence groups in the RUNNING state.
        self.running: List[SequenceGroup] = []
        self.running_stay: List[SequenceGroup] = []
        
        # Sequence groups in the SWAPPED state.
        self.swapped: List[SequenceGroup] = []
        self.prefilled: List[SequenceGroup] = []
        
        self.last_logging_time: float = 0.0
        # List[timestamp, num_tokens]
        self.num_input_tokens: List[Tuple[float, int]] = []

        #self.max_running_seq_len: int = 0
        #self.re_compute: int = 0
        #self.re_swap: int = 0
        #self.ite: int = 0
        #self.expelled: int = 0

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: str) -> None:
        for state_queue in [self.waiting, self.running, self.swapped]:
            for seq_group in state_queue:
                if seq_group.request_id == request_id:
                    # Remove the sequence group from the state queue.
                    state_queue.remove(seq_group)
                    for seq in seq_group.seqs:
                        if seq.is_finished():
                            continue
                        self.free_seq(seq, SequenceStatus.FINISHED_ABORTED)
                    return

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped or self.running_stay

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def covert_prefilled_to_running(self):
        while self.prefilled:
            seq_group = self.prefilled.pop(0)
            for seq in seq_group.get_seqs():
                seq.status = SequenceStatus.RUNNING
            self.running.append(seq_group)
        # self.running.sort(key=lambda x:int(len(x.seqs[0].prompt)))
            
    def covert_running_to_prefilled(self):
        while self.running:
            seq_group = self.running.pop(0)
            for seq in seq_group.get_seqs():
                seq.status = SequenceStatus.PREFILLED
            self.prefilled.append(seq_group)
            # print(f"req {seq_group.request_id} is finished prefill ", time.time())

    def covert_prefilled_to_running_stay(self):
        while self.prefilled:
            seq_group = self.prefilled.pop(0)
            for seq in seq_group.get_seqs():
                seq.status = SequenceStatus.RUNNING
            self.running_stay.append(seq_group)
        #self.running_stay.sort(key = lambda x: x.resoucre_need)

    '''def calculateNeed(self,
                      need: List[int], 
                      max_need: List[int], 
                      allocate: List[int]) -> None:
        for i in range(len(max_need)):
            need[i] = max_need[i] - allocate[i]

    def is_safe(self, 
                seqs: int, 
                available: int, 
                max_need: List[int], 
                allocate: List[int]) -> bool:
        need = [0] * seqs
        self.calculateNeed(need, max_need, allocate)

        finish = [0] * seqs
        work = available

        count = 0
        while count < seqs:
            found = False
            for p in range(seqs):
                if finish[p] == 0:
                    can = True
                    if need[p] > work:
                        can = False
                    if can:
                        work += allocate[p]
                        count += 1
                        finish[p] = 1
                        found = True
            if found == False:
                return False
        
        return True'''

    def _schedule(self, banker: Optional[bool] = False) -> Tuple[SchedulerOutputs, List[str], List[SequenceGroup]]:
        #self.ite += 1

        # Blocks that need to be swaped or copied before model execution.
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}
        ignored_seq_groups: List[SequenceGroup] = []

        # Fix the current time.
        now = time.time()
        
        # NOTE(woosuk): We prioritize the sequence groups in the RUNNING state
        # in order to minimize the preemption overheads.
        # Preemption happens only when there is no available slot to keep all
        # the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        
        '''if len(self.running) == 0 and len(self.running_stay) != 0:
            running: List[SequenceGroup] = []
            total_resource = self.block_manager.free_tokens()
            self.running_stay.sort(key = lambda x: x.resoucre_need)
            
            # must can
            seq_group = self.running_stay.pop()
            running.append(seq_group)
            total_resource -= seq_group.seqs[0].data.get_len()

            length = len(self.running_stay)
            count = 0    
            while self.running_stay:
                seq_group = self.running_stay[0]
                if seq_group.seqs[0].data.get_len() <= total_resource:
                    input_seq_group = self.running_stay.pop(0)
                    total_resource -= input_seq_group.seqs[0].data.get_len()
                    running.append(input_seq_group)
                count += 1
                if count == length:
                    break
            
            self.running = running
        
        elif len(self.running) != 0 and len(self.running_stay) != 0:
            extend_running: List[SequenceGroup] = []

            num_batched_tokens = sum(seq_group.seqs[0].data.get_len() for seq_group in self.running)
            available = self.scheduler_config.max_num_batched_tokens - num_batched_tokens
            
            max_need = []
            for seq_group in self.running:
                max_need.append(seq_group.resoucre_need)
            allocate = []
            for seq_group in self.running:
                allocate.append(seq_group.seqs[0].data.get_len())
            cur_max_tokens = -1
            for seq_group in self.running:
                cur_max_tokens = max(cur_max_tokens, seq_group.resoucre_need)
            
            self.running_stay.sort(key = lambda x: x.resoucre_need)
            length = len(self.running_stay)
            count = 0
            while self.running_stay:
                seq_group = self.running_stay[0]
                if seq_group.seqs[0].get_len() >= cur_max_tokens:
                    continue
                
                max_need.append(seq_group.resoucre_need)
                allocate.append(0)
                if self.is_safe(seqs = len(self.running) + 1, 
                                available = available,
                                max_need = max_need, 
                                allocate = allocate):
                    intput_seq_group = self.running_stay.pop(0)
                    extend_running.append(intput_seq_group)
                max_need.pop()
                allocate.pop()
                count += 1
                if count == length:
                    break
            
        
            self.running.extend(extend_running)'''
        if banker:
            '''total_free_gpu_blocks = self.block_manager.get_num_free_gpu_blocks()
            future_running: List[SequenceGroup] = []

            for seq_group in self.running:
                for seq in seq_group.get_seqs(status = SequenceStatus.RUNNING):
                    seq_group.used = seq_group.resoucre_need - math.ceil(seq.get_output_len() / self.cache_config.block_size)
            if self.running:
                self.running.sort(key = lambda x: x.used)
           
            while True:
                if not self.running:
                    break
                if len(self.running) * self.running[0].used <= total_free_gpu_blocks:
                    break
                else:
                    seq_group = self.running.pop(-1)
                    future_running.append(seq_group)
            
            future_running_stay: List[SequenceGroup] = []
            for seq_group in self.running_stay:
                for seq in seq_group.get_seqs(status = SequenceStatus.RUNNING):
                    seq_group.used = seq_group.resoucre_need - math.ceil(seq.get_output_len() / self.cache_config.block_size)
            if self.running_stay:
                self.running_stay.sort(key = lambda x: x.used)

            if self.running:
                cur_min = self.running[0].used
            else:
                cur_min = 1 << 30
            while self.running_stay:
                seq_group = self.running_stay.pop(0)
                temp = cur_min
                cur_min = min(cur_min, seq_group.used)
                if cur_min * (len(self.running) + 1) <= total_free_gpu_blocks:
                    self.running.append(seq_group)
                else:
                    future_running_stay.append(seq_group)
                    cur_min = temp

            self.running_stay.extend(future_running_stay)
            self.running_stay.extend(future_running)'''
                
            length_runnging_stay = len(self.running_stay)
            #length_running = len(self.running)
            #temp_running = self.running.copy()
            #temp_running_stay = self.running_stay.copy()
            '''cur_max = -1
            if length_running != 0:
                #temp_running.sort(key = lambda x: x.predicted_len)
                temp_running.sort(key = lambda x: x.resoucre_need)
                #cur_max = max(cur_max, temp_running[-1].predicted_len)
                cur_max = max(cur_max, temp_running[-1].resoucre_need)
            if cur_max != -1:
                if cur_max != self.max_running_seq_len:
                    add_long = True
                else:
                    add_long = False
            else:
                add_long = True'''
           
            #total_blocks = self.cache_config.num_gpu_blocks
            total_free_tokens = self.block_manager.get_num_free_gpu_blocks()
            min_resource_need = []
            if length_runnging_stay != 0:
                backup: List[SequenceGroup] = []
                #self.running_stay.sort(key = lambda x: x.resoucre_need)
                #total_free_tokens = self.block_manager.get_num_free_gpu_blocks() * self.cache_config.block_size
                # total_free_tokens = self.block_manager.get_num_free_gpu_blocks()
                # min_resource_need = []
                
                for seq_group in self.running:
                    for temp_run_seq in seq_group.get_seqs(status = SequenceStatus.RUNNING):
                        t = seq_group.resoucre_need - math.ceil(temp_run_seq.get_output_len() / 16)
                        #t = seq_group.resoucre_need - len(temp_run_seq.logical_token_blocks)
                        #t = seq_group.resoucre_need
                        if t > 0:
                            min_resource_need.append(t)
                        else:
                            print(f"In running: add is not over 0!")
                while self.running_stay:
                    #if add_long:
                    #    seq_group = self.running_stay[-1]
                    #    for temp_run_seq in seq_group.get_seqs(status = SequenceStatus.RUNNING):
                    #        min_resource_need.append(seq_group.resoucre_need - temp_run_seq.get_len())
                    #    if min(min_resource_need) * len(min_resource_need) <= total_free_tokens:
                    #        input = self.running_stay.pop()
                    #        temp_running.append(input)
                    #    else:
                    #        min_resource_need.pop()
                    #    add_long = False
                    #    count += 1
                    
                    seq_group = self.running_stay.pop(0)
                    add = False
                    for temp_run_seq in seq_group.get_seqs(status = SequenceStatus.RUNNING):
                        t = seq_group.resoucre_need - math.ceil(temp_run_seq.get_output_len() / 16)
                        #t = seq_group.resoucre_need - len(temp_run_seq.logical_token_blocks)
                        #t = seq_group.resoucre_need
                        if t > 0:
                            min_resource_need.append(t)
                            add = True
                    if add:
                        if min(min_resource_need) * len(min_resource_need) <= self.block_manager.num_total_gpu_blocks:
                        # if sum(min_resource_need) <= total_free_tokens:
                            self.running.append(seq_group)
                            # print("add resource need ", min(min_resource_need) * len(min_resource_need), total_free_tokens)
                            # print(f"min is {min(min_resource_need)}, length is {len(min_resource_need)}", "total blocks is {total_free_tokens}")
                        else:
                            min_resource_need.pop(-1)
                            backup.append(seq_group)
                    else:
                        backup.append(seq_group)
                        print(f"In running_stay: add is not over 0!")

                #temp_running.sort(key = lambda x: x.resoucre_need, reverse = True)
                #self.max_running_seq_len = temp_running[0].resoucre_need
                #self.running = temp_running.copy()
                self.running_stay = backup
                
            '''if length_runnging_stay != 0:
                self.running_stay.sort(key = lambda x: x.predicted_len)
                count = 0
                #total_free_gpu_blocks = self.block_manager.num_total_gpu_blocks
                total_free_tokens = self.block_manager.num_total_gpu_blocks * self.cache_config.block_size
                while self.running_stay:
                    #total_used_gpu_blocks = 0
                    total_used_tokens = 0
                    for temp_run in temp_running:
                        for temp_run_seq in temp_run.get_seqs(status = SequenceStatus.RUNNING):
                            #total_used_gpu_blocks += len(temp_run_seq.logical_token_blocks)
                            total_used_tokens += temp_run_seq.get_len()
                    resource_need = []
                    for temp_run in temp_running:
                        for temp_run_seq in temp_run.get_seqs(status = SequenceStatus.RUNNING):
                            #resource_need.append(temp_run.resoucre_need - len(temp_run_seq.logical_token_blocks))
                            resource_need.append(temp_run.resoucre_need - temp_run_seq.get_len())
                    
                    if add_long:
                        seq_group = self.running_stay[-1]
                        cur_resoucre_need = 0
                        for cur_req_seq in seq_group.get_seqs(status = SequenceStatus.RUNNING):
                            #cur_resoucre_need += len(cur_req_seq.logical_token_blocks)
                            cur_resoucre_need += cur_req_seq.get_len()
                        for cur_req_seq in seq_group.get_seqs(status = SequenceStatus.RUNNING):
                            #resource_need.append(seq_group.resoucre_need - len(cur_req_seq.logical_token_blocks))
                            resource_need.append(seq_group.resoucre_need - cur_req_seq.get_len())
                        min_resource_need = min(resource_need)
                        future_min_resource_need = min_resource_need * len(resource_need)
                        #if total_used_gpu_blocks +   cur_resoucre_need + future_min_resource_need <= total_free_gpu_blocks:
                        if total_used_tokens + cur_resoucre_need + future_min_resource_need <= total_free_tokens:
                            input = self.running_stay.pop()
                            temp_running.append(input)
                            #total_free_gpu_blocks -= total_used_gpu_blocks +   cur_resoucre_need + future_min_resource_need
                            total_used_tokens += cur_resoucre_need
                        else:
                            resource_need.pop()
                        add_long = False
                        count += 1
                    
                    seq_group = self.running_stay[0]
                    cur_resoucre_need = 0
                    for cur_req_seq in seq_group.get_seqs(status = SequenceStatus.RUNNING):
                        #cur_resoucre_need += len(cur_req_seq.logical_token_blocks)
                        cur_resoucre_need += cur_req_seq.get_len()
                    for cur_req_seq in seq_group.get_seqs(status = SequenceStatus.RUNNING):
                        #resource_need.append(seq_group.resoucre_need - len(cur_req_seq.logical_token_blocks))
                        resource_need.append(seq_group.resoucre_need - cur_req_seq.get_len())
                    min_resource_need = min(resource_need)
                    future_min_resource_need = min_resource_need * len(resource_need)
                    #if total_used_gpu_blocks +   cur_resoucre_need + future_min_resource_need <= total_free_gpu_blocks:
                    if total_used_tokens + cur_resoucre_need + future_min_resource_need <= total_free_tokens:
                        input = self.running_stay.pop(0)
                        temp_running.append(input)
                        #total_free_gpu_blocks -= total_used_gpu_blocks +   cur_resoucre_need + future_min_resource_need
                        total_used_tokens += cur_resoucre_need
                    else:
                        resource_need.pop()
                    count += 1
                    
                    if count == length_runnging_stay:
                        break
            
            temp_running.sort(key = lambda x: x.predicted_len)
            self.max_running_seq_len = temp_running[-1].predicted_len
            self.running = temp_running.copy()'''
            #if min_resource_need:
            #    if min(min_resource_need) * len(min_resource_need) > total_free_tokens:
            #        with open("/workspace/vllm/benchmarks/over.txt", 'a') as file:
            #            file.write(f"In ite {self.ite}, {min(min_resource_need) * len(min_resource_need)}, {total_free_tokens}\n")
                        #print("resource info " ,self.ite , min(min_resource_need) * len(min_resource_need), total_free_tokens)

        if banker is False:               
            self.running = self.policy.sort_by_priority(now, self.running)    
            # self.running = self.policy.sort_by_priority(now, self.running)    
            # self.running.sort(key=lambda x:int(len(x.seqs[0].data.output_token_ids)),reverse=True)
            # self.running.sort(key=lambda x:int(len(x.seqs[0].data.prompt_token_ids)))

        # Reserve new token slots for the running sequence groups.
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        #ite = 0
        '''t_expelled = 0
        used_blocks = 0
        for seq_group in self.running:
            for seq in seq_group.get_seqs(status = SequenceStatus.RUNNING):
                used_blocks += len(seq.logical_token_blocks)
        free_blocks = self.block_manager.get_num_free_gpu_blocks()'''
        
        #if banker is False:
        #if True:
        while self.running:
                #t1 = self.block_manager.get_num_free_gpu_blocks()

            seq_group = self.running.pop(0)
                
            while not self.block_manager.can_append_slot(seq_group):
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    
                    # victim_seq_group = self.running.pop(-1)
                    victim_seq_group = self.running_stay.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                    #self.expelled += 1
                    #t_expelled += 1
                    #print(f"In ite {self.ite} this req has been expelled from running queue total {self.expelled}")
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group) 
                    #self.expelled += 1
                    #t_expelled += 1
                    #print(f"In ite {self.ite} this req has been expelled from running queue total {self.expelled}")
                    break
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq_group, blocks_to_copy)
                running.append(seq_group)
                #label = seq_group.resoucre_need
                #for seq in seq_group.get_seqs(status = SequenceStatus.RUNNING):
                #    label += len(seq.get_token_ids())
                #self.max_running_seq_len = max(self.max_running_seq_len, label)
        '''if t_expelled != 0:
            with open("/workspace/vllm/benchmarks/expelled.txt", 'a') as file:
                file.write(f"In ite {self.ite}, {t_expelled} seqs has been expelled\n")
                file.write(f"{used_blocks} blocks have been allocated while {free_blocks} blocks are free\n")'''

            #t2 = self.block_manager.get_num_free_gpu_blocks()
            #with open("/workspace/vllm/benchmarks/blocks.txt", 'a') as file:
            #    file.write(f"befor ite {ite} has {t1} blocks, after it has {t2} blocks\n")
            #ite += 1

        self.running = running
        
        #else:
        #    while self.running:
        #        seq_group = self.running.pop(0)

        #        resource_need = 0
        #        for seq in seq_group.get_seqs(status = SequenceStatus.RUNNING):
        #            resource_need += seq_group.resoucre_need - math.ceil(seq.get_output_len() / 16)
                
        #        if resource_need <= self.block_manager.get_num_free_gpu_blocks():
        #            self._append_slot(seq_group, blocks_to_copy)
        #            running.append(seq_group)
        #        else:
        #            self._preempt(seq_group, blocks_to_swap_out)
        #            preempted.append(seq_group) 
            
        #    self.running = running

        # Swap in the sequence groups in the SWAPPED state if possible.
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        while self.swapped and not blocks_to_swap_out:
            seq_group = self.swapped[0]
            # If the sequence group has been preempted in this step, stop.
            if seq_group in preempted:
                #print(f"In swap: this swapped req has been preempted")
                break
            # If the sequence group cannot be swapped in, stop.
            if not self.block_manager.can_swap_in(seq_group):
                #print(f"In swap: can't swap in no enough blocks")
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
            num_curr_seqs = sum(
                seq_group.num_seqs(status=SequenceStatus.RUNNING)
                for seq_group in self.running)
            if (num_curr_seqs + num_new_seqs >
                    self.scheduler_config.max_num_seqs):
                #print(f"add too more swapped req into running queue")
                break

            seq_group = self.swapped.pop(0)
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slot(seq_group, blocks_to_copy)
            self.running.append(seq_group)
            #label = seq_group.resoucre_need
            #for seq in seq_group.get_seqs(status = SequenceStatus.RUNNING):
            #    label += len(seq.get_token_ids())
            #self.max_running_seq_len = max(self.max_running_seq_len, label)

        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running)

        # Join waiting sequences if possible.
        prompt_group_ids: List[str] = []
        # NOTE(woosuk): The sequence groups in the SWAPPED state are strictly
        # prioritized over the sequence groups in the WAITING state.
        # This is because we want to bound the amount of CPU memory taken by
        # the swapped sequence groups.
        if not self.swapped:
            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            while self.waiting:
                seq_group = self.waiting[0]
                # If the sequence group has been preempted in this step, stop.
                if seq_group in preempted:
                    #print(f"In waiting: this waiting queue has been preempted")
                    break

                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                if num_prompt_tokens >= self.scheduler_config.max_seq_len:
                    # print("no space 1")
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        " and exceeds limit of "
                        f"{self.scheduler_config.max_seq_len}")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    break

                # If the sequence group cannot be allocated, stop.
                if not self.block_manager.can_allocate(seq_group):
                    #print(f"In waiting: there is no enough blocks")
                    break

                # If the number of batched tokens exceeds the limit, stop.
                if (num_batched_tokens + num_prompt_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    #print(f"In waiting: more than max_num_batched_tokens")
                    break

                # The total number of sequences in the RUNNING state should not
                # exceed the maximum number of sequences.
                num_new_seqs = seq_group.num_seqs(
                    status=SequenceStatus.WAITING)
                num_curr_seqs = sum(
                    seq_group.num_seqs(status=SequenceStatus.RUNNING)
                    for seq_group in self.running)
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    #print(f"In waiting: more than batch size")
                    break
                seq_group = self.waiting.pop(0)
                self._allocate(seq_group)
                self.running.append(seq_group)
                #label = seq_group.resoucre_need
                #for seq in seq_group.get_seqs(status = SequenceStatus.RUNNING):
                #    label += len(seq.get_token_ids())
                #self.max_running_seq_len = max(self.max_running_seq_len, label)
                num_batched_tokens += num_prompt_tokens
                prompt_group_ids.append(seq_group.request_id)

        scheduler_outputs = SchedulerOutputs(
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
        )
        
        if not self.log_stats:
            return scheduler_outputs, prompt_group_ids, ignored_seq_groups

        # TODO(woosuk): Move the below code to the engine.
        now = time.time()
        if num_batched_tokens > 0:
            self.num_input_tokens.append((now, num_batched_tokens))
        elapsed_time = now - self.last_logging_time
        if elapsed_time > _LOGGING_INTERVAL_SEC:
            self.last_logging_time = now
            self.num_input_tokens = [(t, n) for t, n in self.num_input_tokens
                                     if now - t < _LOGGING_INTERVAL_SEC]
            if len(self.num_input_tokens) > 1:
                total_num_tokens = sum(n
                                       for _, n in self.num_input_tokens[:-1])
                window = now - self.num_input_tokens[0][0]
                avg_throughput = total_num_tokens / window
            else:
                avg_throughput = 0.0

            total_num_gpu_blocks = self.cache_config.num_gpu_blocks
            num_free_gpu_blocks = self.block_manager.get_num_free_gpu_blocks()
            num_used_gpu_blocks = total_num_gpu_blocks - num_free_gpu_blocks
            gpu_cache_usage = num_used_gpu_blocks / total_num_gpu_blocks

            total_num_cpu_blocks = self.cache_config.num_cpu_blocks
            if total_num_cpu_blocks > 0:
                num_free_cpu_blocks = (
                    self.block_manager.get_num_free_cpu_blocks())
                num_used_cpu_blocks = total_num_cpu_blocks - num_free_cpu_blocks
                cpu_cache_usage = num_used_cpu_blocks / total_num_cpu_blocks
            else:
                cpu_cache_usage = 0.0

            logger.info(f"Throughput: {avg_throughput:.1f} tokens/s, "
                        f"Running: {len(self.running)} reqs, "
                        f"Swapped: {len(self.swapped)} reqs, "
                        f"Pending: {len(self.waiting)} reqs, "
                        f"GPU KV cache usage: {gpu_cache_usage * 100:.1f}%, "
                        f"CPU KV cache usage: {cpu_cache_usage * 100:.1f}%")
        return scheduler_outputs, prompt_group_ids, ignored_seq_groups

    # def store_prompt_kv_cache(self):
    #     for seq_group in self.running:
    #         for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
    #             print(" running seq after interation",seq.seq_id)
    #     for seq_group in self.swapped:
    #         for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
    #             print(" swapped seq after interation",seq.seq_id) 
    #     for seq_group in self.waiting:
    #         for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
    #             print(" waiting seq after interation",seq.seq_id) 
                
    def schedule(
        self,
        banker: Optional[bool] = False
    ) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs,
               List[SequenceGroup]]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        (scheduler_outputs, prompt_group_ids,
         ignored_seq_groups) = self._schedule(banker = banker)

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        # print("schedule self running ", len(self.running))
        for seq_group in self.running:
            is_prompt = seq_group.request_id in prompt_group_ids

            seq_data: Dict[int, List[SequenceData]] = {}
            block_tables: Dict[int, List[int]] = {}
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=is_prompt,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
            )
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs, ignored_seq_groups

    def update(
        self,
        seq_outputs: Dict[int, SequenceOutputs],
    ) -> List[SequenceGroup]:
        # Update the running sequences and free blocks.
        for seq_group in self.running:
            # Process beam search results before processing the new tokens.
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                output = seq_outputs[seq.seq_id]
                if seq.seq_id != output.parent_seq_id:
                    # The sequence is a fork of the parent sequence (beam
                    # search). Free the current sequence.
                    self.block_manager.free(seq)
                    # Fork the parent sequence.
                    parent_seq = seq_group.find(output.parent_seq_id)
                    parent_seq.fork(seq)
                    self.block_manager.fork(parent_seq, seq)

            # Process the new tokens.
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                # Append a new token to the sequence.
                output = seq_outputs[seq.seq_id]
                seq.append_token_id(output.output_token, output.logprobs)
        # Return a shallow copy of the running queue to prevent the queue
        # from being modified by the caller.
        return self.running.copy()

    def free_seq(self, seq: Sequence, finish_status: SequenceStatus) -> None:
        seq.status = finish_status
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        # for seq_group in self.running:
        #     print("finished ", seq_group.is_finished())
            
        self.running = [
            seq_group for seq_group in self.running
            if not seq_group.is_finished()
        ]

    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs():
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
        # (e.g., beam search), recomputation is not supported. In such a case,
        # we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        if preemption_mode is None:
            seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
            if len(seqs) == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            assert False, "Invalid preemption mode."

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.block_manager.free(seq)
        # NOTE: For FCFS, we insert the preempted sequence group to the front
        # of the waiting queue.
        self.waiting.insert(0, seq_group)
        #self.re_compute += 1
        #print(f"In ite{self.ite} a seq has been recomputed total {self.re_compute} times")

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        for seq in seqs:
            seq.status = SequenceStatus.SWAPPED
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped.append(seq_group)
        #self.re_swap += 1
        #print(f"In ite{self.ite} a seq has been swapped total {self.re_swap} times")

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
