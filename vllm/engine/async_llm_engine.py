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
from vllm.utils import random_uuid

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


    def add_reuqests_to_mul_generate_hhy(
        self,
        output_lens: Optional[List[int]],
        sampling_params: Optional[List[SamplingParams]],
        prompts_tokens_ids: Optional[List[List[int]]]) -> None:

        arrival_time = time.time()
        for prompt_tokens_ids, sampling_param, output_len in zip(prompts_tokens_ids, sampling_params, output_lens):
            request_id = random_uuid()
            sampling_param.max_tokens = int(output_len)
            if self.engine_use_ray:
                self.engine.add_request.remote(
                    request_id = request_id,
                    sampling_params = sampling_param,
                    prompt_token_ids = prompt_tokens_ids,
                    arrival_time = arrival_time
                )
            else:
                self.engine.add_request(
                    request_id = request_id,
                    sampling_params = sampling_param,
                    prompt_token_ids = prompt_tokens_ids,
                    arrival_time = arrival_time
                )

    def mul_generate_hhy(self) -> RequestOutput:
        
        start_time = time.time()
        outputs: List[RequestOutput] = []
        while self.engine.has_unfinished_requests():
            step_outputs = self.engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
        end_time = time.time()
        total_num_tokens = sum(len(output.prompt_token_ids) + len(output.outputs[0].token_ids) for output in outputs)
        total_num_reqs = len(outputs)
        print(f"Processed {total_num_tokens} tokens in {total_num_reqs} reqs")
        print(f"Throughput: {total_num_tokens / {end_time - start_time}:.2f}/tokens per second")
        print(f"Throughput: {total_num_reqs / {end_time - start_time}:.2f}/reqs per second") 

    def mul_generate(
            self,
            prompts: Optional[List[str]],
            output_lens: Optional[List[int]],
            sampling_params: List[SamplingParams],
            prompt_token_ids: Optional[List[List[int]]] = None) -> RequestOutput:
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
        # start_add_request_time = time.time()
        # Create an event to notify us that there is new output from the
        # vLLM engine.
        for prompt, output_len, sampling_param in zip(prompts, output_lens,sampling_params):
            request_id = random_uuid()
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
        # end_add_request_time = time.time()
        # print("start_add_request_time, end_add_request_time ", start_add_request_time, end_add_request_time)
        start = time.time()
        #prefill_execute_time = 0 
        outputs: List[RequestOutput] = []
        while self.engine.has_unfinished_requests():
            step_outputs = self.engine.step()
            #if prefill_execute_time == 0:
            #    prefill_execute_time = time.time()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    # print("output: ", output )
        end = time.time()

        elapsed_time = end-start
        total_num_tokens = sum(
            len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
            for output in outputs
        )
        print("arrival_time, start time, end time , total_num_tokens, outputs len ", arrival_time, start, end, total_num_tokens, len(outputs))

        print(f"Throughput: {len(outputs) / elapsed_time:.2f} requests/s, "
            f"{total_num_tokens / elapsed_time:.2f} tokens/s")


       
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
            parallel_config=parallel_config, engine_use_ray=engine_args.engine_use_ray)
        # Create the async LLM engine.
        engine = cls(engine_args.worker_use_ray,
                     engine_args.engine_use_ray,
                     *engine_configs,
                     distributed_init_method,
                     devices,
                     log_requests=not engine_args.disable_log_requests,
                     log_stats=not engine_args.disable_log_stats)
        return engine
