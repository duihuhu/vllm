import asyncio
import time
from typing import Dict, List, Optional

from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.ray_utils import initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
import copy
logger = init_logger(__name__)

TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds


class AsyncLLMEngine:
    """An asynchronous wrapper for LLMEngine.

    This class is used to wrap the LLMEngine class to make it asynchronous. It
    uses asyncio to create a background loop that keeps processing incoming
    requests. The LLMEngine is kicked by the generate method when there
    are requests in the waiting queue. The generate method yields the outputs
    from the LLMEngine to the caller.

    NOTE: For the comprehensive list of arguments, see `LLMEngine`.

    Args:
        worker_use_ray: Whether to use Ray for model workers. Required for
            distributed execution. Should be the same as
            `parallel_config.worker_use_ray`.
        engine_use_ray: Whether to make LLMEngine a Ray actor. If so, the
            async frontend will be executed in a separate process as the
            model workers.
        log_requests: Whether to log the requests.
        *args, *kwargs: Arguments for LLMEngine.
    """

    def __init__(self,
                 worker_use_ray: bool,
                 engine_use_ray: bool,
                 *args,
                 log_requests: bool = True,
                 **kwargs) -> None:
        self.worker_use_ray = worker_use_ray
        self.engine_use_ray = engine_use_ray
        self.log_requests = log_requests
        if not self.engine_use_ray:
            engine_class = LLMEngine
        elif self.worker_use_ray:
            engine_class = ray.remote(num_cpus=0)(LLMEngine).remote
        else:
            engine_class = ray.remote(num_gpus=1)(LLMEngine).remote
        self.engine = engine_class(*args, **kwargs)
        # Request id -> request output.
        self.request_outputs: Dict[str, RequestOutput] = {}
        # Request id -> event to notify that there is new output.
        self.request_events: Dict[str, asyncio.Event] = {}
        self.is_engine_running = False
        self.kicking_request_id: Optional[str] = None

        self.resued_request_ids = []
        self.total_num_requests = 0
    async def engine_step(self, kicking_request_id: Optional[str] = None):
        """Kick the engine to process the waiting requests."""
        self.is_engine_running = True
        self.kicking_request_id = kicking_request_id
        if self.engine_use_ray:
            request_outputs = await self.engine.step.remote()
        else:
            # Yield to the event loop to allow other coroutines to run
            # while is_engine_running is True. This let the engine to add new
            # requests into the queue.
            await asyncio.sleep(0)
            request_outputs = self.engine.step()
        self.is_engine_running = False
        self.kicking_request_id = None

        # Notify the waiting coroutines that there are new outputs ready.
        for request_output in request_outputs:
            request_id = request_output.request_id
            self.request_outputs[request_id] = request_output
            self.request_events[request_id].set()

    def convert_reqs_status(self,request_ids: List[str]):
        self.engine.convert_reqs_status(request_ids)
        
    def convert_req_label_status(self,request_id, label, arrive_time):
        self.engine.convert_req_label_status(request_id, label, arrive_time)

    def convert_req_label_status_dict(self,request_id, label, arrive_time):
        self.engine.convert_req_label_status_dict(request_id, label, arrive_time)
        
    def convert_reqs_status_by_num(self,request_num):
        self.engine.convert_reqs_status_by_num(request_num)

    def copy_reused_seqs_info(self):
        resued_time = len(self.resued_request_ids) / self.total_num_requests
        for i in range(resued_time):
            for seq_group in self.engine.scheduler.prefilled:
                seqs: List[Sequence] = []    
                for seq in seq_group.get_seqs():                
                    arrival_time = time.time()
                    sampling_params = SamplingParams(
                        n=1,
                        temperature=seq_group.sampling_params.temperature,
                        top_p=1.0,
                        use_beam_search=seq_group.sampling_params.use_beam_search,
                        ignore_eos=True,
                        max_tokens=seq_group.sampling_params.max_tokens
                    )
                    block_size = self.engine.cache_config.block_size
                    for _ in range(sampling_params.best_of):
                        seq_id = next(self.engine.seq_counter)
                        seq1 = Sequence(seq_id, seq.prompt, seq.data.prompt_token_ids, block_size)
                        seq1.data.output_token_ids = copy.deepcopy(seq.prefill_data.output_token_ids)
                        seq1.data.cumulative_logprob = seq.prefill_data.cumulative_logprob
                        seq1.output_logprobs = copy.deepcopy(seq.prefill_output_logprobs)
                        seq1.output_tokens = copy.deepcopy(seq.prefill_output_tokens)
                        seq1.logical_token_blocks.clear()
                        seq1.output_logprobs = seq.prefill_output_logprobs
                        seq1.output_text = seq.prefill_output_text
                        seq1.logical_token_blocks = copy.deepcopy(seq.prefill_logical_token_blocks)
                        seq1.prefill_block_table_number = copy.deepcopy(seq.prefill_block_table_number)
                        seq1.status = SequenceStatus.PREFILLED
                        seqs.append(seq1)
                        self.engine.scheduler.block_manager.copy_block_tables(seq.seq_id, seq_id)
                request_id = self.resued_request_ids.pop(0)
                seq_group1 = SequenceGroup(request_id, seqs, sampling_params, arrival_time)
                self.engine.scheduler.add_to_waiting_prefilled(seq_group1)
        while self.engine.scheduler.prefilled_resued:
            seq_group = self.engine.scheduler.prefilled_resued.pop(0)
            self.engine.scheduler.prefilled.append(seq_group)
            
        for seq_group in self.engine.scheduler.prefilled:
            self.engine.scheduler.prefilled_dict[seq_group.request_id] = seq_group
            
    def generate_mdecode_prefill(self, p_iteration_num):
        while self.engine.has_unfinished_requests():
            step_outputs = self.engine.step()
            # prefilled_num = self.engine.covert_running_to_prefilled()
            out_request_ids = [ output.request_id for output in step_outputs]
            self.engine.convert_outputs_reqs_status(out_request_ids)
            prefilled_num = len(out_request_ids)
            if p_iteration_num + prefilled_num >= self.total_num_requests:
                self.copy_reused_seqs_info()
            self.engine.send_mdecode_prefilled_controller(prefilled_num)
            return p_iteration_num + prefilled_num
            # print("mdecode!!: complish mdecode prefill request ", prefilled_num)
            
    def generate_decode(self):
        outputs: List[RequestOutput] = []
        s_time = time.time()
        while self.engine.has_unfinished_decode_requests():
            # print("mdecode decode iteration ", self.engine.get_num_unfinished_requests())
            # start_time =  time.time()
            step_outputs = self.engine.mdecode_step()
            # end_time =  time.time()
            # print("iteration time ", end_time-start_time)
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    end_time = time.time()
                    print("decode complish ", output.request_id, end_time, end_time-s_time, output.outputs[0].finish_reason)
                    
    def add_mprefill_request(self,
        prompts: Optional[List[str]],
        output_lens: Optional[List[int]],
        request_ids: Optional[List[str]],
        sampling_params: List[SamplingParams],
        prompt_token_ids: Optional[List[List[int]]] = None):
            arrival_time = time.time()
            for prompt, request_id, output_len, sampling_param in zip(prompts, request_ids, output_lens, sampling_params):
                sampling_param.max_tokens = int(output_len)
                if self.engine_use_ray:
                    self.engine.add_mprefill_request.remote(
                        request_id,
                        prompt,
                        sampling_param,
                        prompt_token_ids=prompt_token_ids,
                        arrival_time=arrival_time)
                else:
                    self.engine.add_mprefill_request(request_id,
                                            prompt,
                                            sampling_param,
                                            prompt_token_ids=prompt_token_ids,
                                            arrival_time=arrival_time)        
    def add_reused_request_ids(self, resued_request_ids: List[str]):
        for request_id in resued_request_ids:
            self.resued_request_ids.append(request_id)
            
    def add_request(self,
        prompts: Optional[List[str]],
        output_lens: Optional[List[int]],
        request_ids: Optional[List[str]],
        sampling_params: List[SamplingParams],
        prompt_token_ids: Optional[List[List[int]]] = None):
            arrival_time = time.time()
            for prompt, request_id, output_len, sampling_param in zip(prompts, request_ids, output_lens, sampling_params):
                # if self.log_requests:
                #     logger.info(f"Received request {request_id}: "
                #                 f"prompt: {prompt!r}, "
                #                 f"sampling params: {sampling_params}, "
                #                 f"prompt token ids: {prompt_token_ids}.")

                # Add the request into the vLLM engine's waiting queue.
                sampling_param.max_tokens = int(output_len)
                if self.engine_use_ray:
                    self.engine.add_request.remote(
                        request_id,
                        prompt,
                        sampling_param,
                        prompt_token_ids=prompt_token_ids,
                        arrival_time=arrival_time)
                else:
                    self.engine.add_request(request_id,
                                            prompt,
                                            sampling_param,
                                            prompt_token_ids=prompt_token_ids,
                                            arrival_time=arrival_time)

    def mprefill_generate_prefill(
        self, mm, prefill_nums) -> int:
        while self.engine.has_unfinished_prefill_requests():
            # print("mprefill prefill iteration")
            self.engine.move_waitingadd_to_waiting()
            step_outputs = self.engine.step()
            
            out_request_ids = [ output.request_id for output in step_outputs]
            # self.engine.send_mprefilled_to_mdecode(out_request_ids)
            self.engine.convert_outputs_reqs_status(out_request_ids)
            #write request num to 
            prefill_nums = prefill_nums + 1
            request_num = len(step_outputs)
            # print("mprefill_generate_prefill: " ,prefill_nums, len(step_outputs), time.time())

            combined_info_bytes = prefill_nums.to_bytes(1, byteorder='big') + request_num.to_bytes(1, byteorder='big')
            mm.seek(0)
            mm.write(combined_info_bytes)
            return prefill_nums
            
        print("mprefill!!:  prefill iteration now is no unfinished")
    
    def generate_prefill(
        self) -> RequestOutput:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.

        Yields:
            The output `RequestOutput` objects from the LLMEngine for the
            request.
        """

        # Preprocess the request.
        # vLLM engine.
        # if status == 'init_mdecode_prefill':
        #     # outputs: List[RequestOutput] = []
        #     while self.engine.has_unfinished_requests():
        #         # print("mdecode prefill iteration")
        #         step_outputs = self.engine.step()
        #         prefilled_num = self.engine.covert_running_to_prefilled()
        #         self.engine.send_mdecode_prefilled_controller(prefilled_num)
            # print("already complish prefill request ")
        # if status == 'mprefill_execute':
        while self.engine.has_unfinished_requests():
            # print("mprefill prefill iteration")
            step_outputs = self.engine.step()
            out_request_ids = [ output.request_id for output in step_outputs]
            self.engine.send_mprefilled_to_mdecode(out_request_ids)
        print("mprefill!!:  prefill iteration now is no unfinished")
            #todo list
            # print("todo ")
     
    def monitor_mdecode_info(self):
        return self.engine.monitor_mdecode_info()   
    
    async def generate(
            self,
            prompt: Optional[str],
            sampling_params: SamplingParams,
            request_id: str,
            prompt_token_ids: Optional[List[int]] = None) -> RequestOutput:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.

        Yields:
            The output `RequestOutput` objects from the LLMEngine for the
            request.
        """
        # Preprocess the request.
        arrival_time = time.time()

        # Create an event to notify us that there is new output from the
        # vLLM engine.
        request_event = asyncio.Event()
        self.request_events[request_id] = request_event

        if self.log_requests:
            logger.info(f"Received request {request_id}: "
                        f"prompt: {prompt!r}, "
                        f"sampling params: {sampling_params}, "
                        f"prompt token ids: {prompt_token_ids}.")

        # Add the request into the vLLM engine's waiting queue.
        if self.engine_use_ray:
            await self.engine.add_request.remote(
                request_id,
                prompt,
                sampling_params,
                prompt_token_ids=prompt_token_ids,
                arrival_time=arrival_time)
        else:
            self.engine.add_request(request_id,
                                    prompt,
                                    sampling_params,
                                    prompt_token_ids=prompt_token_ids,
                                    arrival_time=arrival_time)

        # The vLLM engine does not have a background loop that keeps
        # processing incoming requests. Therefore, we need to keep kicking
        # the engine to process the requests.
        while True:
            if request_id not in self.request_events:
                # The request has been aborted.
                return

            # Kick the engine if the engine is not running.
            if not self.is_engine_running:
                try:
                    await self.engine_step(request_id)
                except RuntimeError as e:
                    await self.abort(request_id)
                    raise e

            # Wait for new output. The group_event will be set in engine_step
            # when there is new output available for the sequence group.
            # Added a timeout to prevent deadlock.
            try:
                await asyncio.wait_for(request_event.wait(),
                                       timeout=TIMEOUT_TO_PREVENT_DEADLOCK)
            except asyncio.TimeoutError:
                continue
            # Reset the event to wait for the next output.
            request_event.clear()

            # Decode and return new outputs.
            request_output = self.request_outputs[request_id]
            yield request_output

            # Once finished, release the resources of the sequence group.
            if request_output.finished:
                if self.log_requests:
                    logger.info(f"Finished request {request_id}.")

                del self.request_outputs[request_id]
                del self.request_events[request_id]
                # Kick the engine if the engine is not running. This is to
                # prevent that there are still requests in engine's waiting
                # queue to be executed.
                if not self.is_engine_running:
                    await self.engine_step()
                break

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if request_id not in self.request_events:
            # The request has already finished or been aborted.
            return

        if self.log_requests:
            logger.info(f"Aborted request {request_id}.")

        if self.engine_use_ray:
            await self.engine.abort_request.remote(request_id)
        else:
            self.engine.abort_request(request_id)

        if request_id in self.request_events:
            del self.request_events[request_id]
        if request_id in self.request_outputs:
            del self.request_outputs[request_id]

        # To prevent deadlock when a request is aborted while the engine is
        # running.
        if self.kicking_request_id == request_id:
            self.is_engine_running = False
            self.kicking_request_id = None

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.engine.get_model_config.remote()
        else:
            return self.engine.get_model_config()

    @classmethod
    def from_engine_args(cls,
                         engine_args: AsyncEngineArgs) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        # Initialize the cluster.
        distributed_init_method, devices = initialize_cluster(
            parallel_config, engine_args.engine_use_ray)
        # Create the async LLM engine.
        engine = cls(engine_args.worker_use_ray,
                     engine_args.engine_use_ray,
                     *engine_configs,
                     distributed_init_method,
                     devices,
                     log_requests=not engine_args.disable_log_requests,
                     log_stats=not engine_args.disable_log_stats)
        return engine
