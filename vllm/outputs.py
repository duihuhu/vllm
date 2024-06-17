import time
from typing import List, Optional, Union, Dict

from vllm.lora.request import LoRARequest
from vllm.sequence import (PromptLogprobs, RequestMetrics, SampleLogprobs,
                           SequenceGroup, SequenceStatus)


class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index: The index of the output in the request.
        text: The generated output text.
        token_ids: The token IDs of the generated output text.
        cumulative_logprob: The cumulative log probability of the generated
            output text.
        logprobs: The log probabilities of the top probability words at each
            position if the logprobs are requested.
        finish_reason: The reason why the sequence is finished.
        stop_reason: The stop string or token id that caused the completion
            to stop, None if the completion finished for some other reason
            including encountering the EOS token.
        lora_request: The LoRA request that was used to generate the output.
    """

    def __init__(
        self,
        index: int,
        text: str,
        token_ids: List[int],
        cumulative_logprob: float,
        logprobs: Optional[SampleLogprobs],
        finish_reason: Optional[str] = None,
        stop_reason: Union[int, str, None] = None,
        lora_request: Optional[LoRARequest] = None,
    ) -> None:
        self.index = index
        self.text = text
        self.token_ids = token_ids
        self.cumulative_logprob = cumulative_logprob
        self.logprobs = logprobs
        self.finish_reason = finish_reason
        self.stop_reason = stop_reason
        self.lora_request = lora_request

    def finished(self) -> bool:
        return self.finish_reason is not None

    def __repr__(self) -> str:
        return (f"CompletionOutput(index={self.index}, "
                f"text={self.text!r}, "
                f"token_ids={self.token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob}, "
                f"logprobs={self.logprobs}, "
                f"finish_reason={self.finish_reason}, "
                f"stop_reason={self.stop_reason})")


class RequestOutput:
    """The output data of a request to the LLM.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
        prompt_token_ids: The token IDs of the prompt.
        prompt_logprobs: The log probabilities to return per prompt token.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
        metrics: Metrics associated with the request.
        lora_request: The LoRA request that was used to generate the output.
    """

    def __init__(
        self,
        request_id: str,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]],
        prompt_logprobs: Optional[PromptLogprobs],
        outputs: Optional[List[CompletionOutput]],
        finished: Optional[bool],
        metrics: Optional[RequestMetrics] = None,
        lora_request: Optional[LoRARequest] = None,
        eprefill_host: Optional[str] = None,
        eprefill_port: Optional[str] = None,
        edecode_host: Optional[str] = None,
        edecode_port: Optional[str] = None,
        is_layer: Optional[bool] = False
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_logprobs = prompt_logprobs
        self.outputs = outputs
        self.finished = finished
        self.metrics = metrics
        self.lora_request = lora_request
        self.eprefill_host = eprefill_host
        self.eprefill_port = eprefill_port
        self.edecode_host = edecode_host
        self.edecode_port = edecode_port
        self.global_ranks: List[int] = None
        self.is_layer = False

    @classmethod
    def from_seq_group(cls, seq_group: SequenceGroup) -> "RequestOutput":
        seqs = seq_group.get_seqs()
        if len(seqs) == 1:
            top_n_seqs = seqs
        else:
            # Get the top-n sequences.
            n = seq_group.sampling_params.n
            if seq_group.sampling_params.use_beam_search:
                sorting_key = lambda seq: seq.get_beam_search_score(
                    seq_group.sampling_params.length_penalty)
            else:
                sorting_key = lambda seq: seq.get_cumulative_logprob()
            sorted_seqs = sorted(seqs, key=sorting_key, reverse=True)
            top_n_seqs = sorted_seqs[:n]

        # Create the outputs.
        # NOTE: We need omit logprobs here explicitly because the sequence
        # always has the logprobs of the sampled tokens even if the
        # logprobs are not requested.
        include_logprobs = seq_group.sampling_params.logprobs is not None
        outputs = [
            CompletionOutput(seqs.index(seq), seq.output_text,
                             seq.get_output_token_ids(),
                             seq.get_cumulative_logprob(),
                             seq.output_logprobs if include_logprobs else None,
                             SequenceStatus.get_finished_reason(seq.status),
                             seq.stop_reason) for seq in top_n_seqs
        ]

        # Every sequence in the sequence group should have the same prompt.
        prompt = seq_group.prompt
        prompt_token_ids = seq_group.prompt_token_ids
        prompt_logprobs = seq_group.prompt_logprobs
        finished = seq_group.is_finished()
        finished_time = time.time() if finished else None
        seq_group.set_finished_time(finished_time)
        return cls(seq_group.request_id,
                   prompt,
                   prompt_token_ids,
                   prompt_logprobs,
                   outputs,
                   finished,
                   seq_group.metrics,
                   lora_request=seq_group.lora_request,
                   eprefill_host=seq_group.edecode_host,
                   eprefill_port=seq_group.eprefill_port,
                   edecode_host=seq_group.edecode_host,
                   edecode_port=seq_group.edecode_port)

    def __repr__(self) -> str:
        return (f"RequestOutput(request_id={self.request_id}, "
                f"prompt={self.prompt!r}, "
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"prompt_logprobs={self.prompt_logprobs}, "
                f"outputs={self.outputs}, "
                f"finished={self.finished}, "
                f"metrics={self.metrics}, "
                f"lora_request={self.lora_request})")

class KvPreparedResponse:
    def __init__(
        self,
        request_id: str,
        error: int,
        error_msg: str,
        computed_blocks: int,
        transfer_tag: str,
        dst_cpu_blocks: Optional[List[int]] = None,
        has_dram: Optional[bool] = False
    ) -> None:
        self.request_id = request_id
        self.error = error
        self.error_msg = error_msg
        self.computed_blocks = computed_blocks
        self.global_ranks = None
        self.transfer_tag = transfer_tag
        self.dst_cpu_blocks = dst_cpu_blocks
        self.has_dram = has_dram
    def __json__(self) -> Dict:
        return {
            "request_id": self.request_id,
            "global_ranks": self.global_ranks,
            "error": self.error,
            "error_msg": self.error_msg,
            "computed_blocks": self.computed_blocks,
            "transfer_tag": self.transfer_tag,
            "dst_cpu_blocks": self.dst_cpu_blocks,
            "has_dram": self.has_dram,
        }

class VLLMLoadInfo:
    def __init__(
        self,
        used_gpu_blocks: int,
        used_cpu_blocks: int,
        remained_gpu_blocks: int,
        remained_cpu_blocks: int,
        num_unfinished_requests: int,
        global_ranks: List[int], 
        timestamp: float
    ) -> None:
        self.used_gpu_blocks = used_gpu_blocks
        self.used_cpu_blocks = used_cpu_blocks
        self.remained_gpu_blocks = remained_gpu_blocks
        self.remained_cpu_blocks = remained_cpu_blocks
        self.num_unfinished_requests = num_unfinished_requests
        self.global_ranks = global_ranks
        self.timestamp = timestamp
        
    def __json__(self):
        return {
            "used_gpu_blocks": self.used_gpu_blocks,
            "used_cpu_blocks": self.used_cpu_blocks,
            "remained_gpu_blocks": self.remained_gpu_blocks,
            "remained_cpu_blocks": self.remained_cpu_blocks,
            "num_unfinished_requests": self.num_unfinished_requests,
            "global_ranks": self.global_ranks,
            "timestamp": self.timestamp
        }


class LayerKvPreparedResponse:
    def __init__(
        self,
        merage_request_id: str,
        computed_blocks: List[int],
        global_ranks: List[int],
        transfer_tag: str,
        is_allocated: List[bool]
    ) -> None:
        self.merage_request_id = merage_request_id
        self.computed_blocks = computed_blocks
        self.global_ranks = global_ranks
        self.transfer_tag = transfer_tag
        self.is_allocated = is_allocated
    def __json__(self) -> Dict:
        return {
            "merage_request_id": self.merage_request_id,
            "computed_blocks": self.computed_blocks,
            "global_ranks": self.global_ranks,
            "transfer_tag": self.transfer_tag,
            "is_allocated": self.is_allocated,

        }
        

class MergeReqInfo:
    def __init__(
        self,
        merage_request_id: str,
        blocks: List[int],
        channel: str,
    ):
        self.merage_request_id = merage_request_id
        self.blocks = blocks
        self.channel = channel
