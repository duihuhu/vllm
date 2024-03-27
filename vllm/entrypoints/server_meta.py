from typing import List, Dict

class InferResults:
    def __init__(
        self,
        request_id,
        opp_ranks,
        prompt_token_ids,
        prefilled_token_id,
        output_logprobs,
        cumulative_logprob,
        sampling_params,
        texts: List[str],
        finished: bool
    ) -> None:
        self.request_id = request_id
        self.opp_ranks = opp_ranks
        self.prompt_token_ids = prompt_token_ids
        self.prefilled_token_id = prefilled_token_id
        self.output_logprobs = output_logprobs
        self.cumulative_logprob = cumulative_logprob
        # self.speculate_token_ids = speculate_token_ids
        self.sampling_params = sampling_params
        self.texts = texts
        self.finished = finished
    
    def __json__(self) -> Dict:
        return {
            "request_id": self.request_id,
            "opp_ranks": self.opp_ranks,
            "prompt_token_ids": self.prompt_token_ids,
            "prefilled_token_id": self.prefilled_token_id,
            "output_logprobs": self.output_logprobs,
            "cumulative_logprob": self.cumulative_logprob,
            # "speculate_token_ids": self.speculate_token_ids,
            "sampling_params": self.sampling_params.__json__(),
            "texts": self.texts,
            "finished": self.finished
        }
    