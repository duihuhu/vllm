from typing import List, Optional, Union
import time

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter

class LLM:
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    NOTE: This class is intended to be used for offline inference. For online
    serving, use the `AsyncLLMEngine` class instead.
    NOTE: For the comprehensive list of arguments, see `EngineArgs`.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        seed: The seed to initialize the random number generator for sampling.
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        seed: int = 0,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            seed=seed,
            **kwargs,
        )
        self.llm_engine = LLMEngine.from_engine_args(engine_args)
        self.request_counter = Counter()

    def get_tokenizer(
            self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer

    def set_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        self.llm_engine.tokenizer = tokenizer

    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[SamplingParams] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        mixed_dummy_prompt_token_ids: Optional[List[List[int]]] = None,
        mixed_sampling_params: Optional[SamplingParams] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        """Generates the completions for the input prompts.

        NOTE: This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: A list of prompts to generate completions for.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
            prompt_token_ids: A list of token IDs for the prompts. If None, we
                use the tokenizer to convert the prompts to token IDs.
            use_tqdm: Whether to use tqdm to display the progress bar.

        Returns:
            A list of `RequestOutput` objects containing the generated
            completions in the same order as the input prompts.
        """
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]
        if prompts is not None and prompt_token_ids is not None:
            if len(prompts) != len(prompt_token_ids):
                raise ValueError("The lengths of prompts and prompt_token_ids "
                                 "must be the same.")
        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        # Add requests to the engine.
        if prompts is not None:
            num_requests = len(prompts)
        else:
            num_requests = len(prompt_token_ids)
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            if prompt_token_ids is None:
                token_ids = None
            else:
                token_ids = prompt_token_ids[i]
            self._add_request(prompt, sampling_params, token_ids)
        
        if mixed_dummy_prompt_token_ids:
            for i in range(len(mixed_dummy_prompt_token_ids)):
                self._add_mix_request(None, mixed_sampling_params, mixed_dummy_prompt_token_ids[i])
                
        return self._run_engine(use_tqdm, split_two_phase=1)

    def _add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]],
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(request_id, prompt, sampling_params,
                                    prompt_token_ids)

    def _add_mix_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]],
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_mix_request(request_id, prompt, sampling_params,
                                    prompt_token_ids)

    def _run_engine(self, use_tqdm: bool, split_two_phase: Optional[int]) -> List[RequestOutput]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests, desc="Processed prompts")
        # Run the engine.
        outputs: List[RequestOutput] = []
        interation = 0
        st = time.time()
        # print(f"Start Prefill at {st}")
        if split_two_phase == 1:
            total_num_token = 0
        iteration_time = []
        while self.llm_engine.has_unfinished_requests():
            #print("interation: ", interation)
            iteration_start = time.time()
            step_outputs = self.llm_engine.step()
            iteartion_end = time.time()
            iteration_time.append(iteartion_end-iteration_start)
            interation = interation  + 1
            if interation == 2 and len(self.llm_engine.scheduler.mixing)!=0:
                self.llm_engine.covert_mixing_to_waiting()
            for output in step_outputs:
                if output.finished:
                    # print(f"req {output.request_id} is finished", len(output.prompt_token_ids), len(output.outputs[0].token_ids), time.time()-st)
                    outputs.append(output)
                    # print(output)
                    if use_tqdm:
                        pbar.update(1)
            if split_two_phase == 1:
                self.llm_engine.covert_running_to_prefilled()
                total_num_token += sum(len(step_output.prompt_token_ids) for step_output in step_outputs)

        # with open("iteration_time.txt", "a+") as fd:
        #     for line in iteration_time:
        #         fd.write(str(line)+'\n')
        # print(f"iteration {iteration_time}")
        if split_two_phase == 1:
            ed = time.time()
            print(f"End Prefill at {ed}", "total prefill time: ", ed-st)
            print(f"Prefill process {total_num_token} tokens")
            print(f"{(total_num_token / (ed-st)):.2f} tokens/s")

        if split_two_phase == 1:
            self.llm_engine.covert_prefilled_to_running()
            st2 = time.time()
            print(f"Start Decode at {st2}")
            interation = 0

            while self.llm_engine.has_unfinished_requests():
                #print("interation: ", interation)
                step_outputs = self.llm_engine.step()
                interation = interation  + 1
                for output in step_outputs:
                    if output.finished:
                        # print(f"req {output.request_id} is finished", len(output.prompt_token_ids), len(output.outputs[0].token_ids), time.time()-st)
                        outputs.append(output)
                        # print(output)
                        if use_tqdm:
                            pbar.update(1)
            ed2 = time.time()
            print(f"iteration {interation}")
            print(f"End Decode at {ed2}", "total decode time: ", ed2-st2)
            total_num_token2 = sum(len(output.outputs[0].token_ids) for output in outputs)
            print(f"Decode process {total_num_token2} tokens")
            print(f"Decode Throughput {(total_num_token2 / (ed2-st2)):.2f} tokens/s")
        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        # print(outputs)   
        with open("iteration_time.txt", "a+") as fd:
                fd.write('\n')         
        return outputs
