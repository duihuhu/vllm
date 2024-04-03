from typing import List, Dict

class InferResults:
    def __init__(
        self,
        request_id,
        opp_ranks,
        prompt_token_ids,
        prompt_logprobs,
        prefilled_token_id,
        output_logprobs,
        cumulative_logprob,
        sampling_params,
        index,
        texts: List[str],
        finished: bool
    ) -> None:
        self.request_id = request_id
        self.opp_ranks = opp_ranks
        self.prompt_token_ids = prompt_token_ids
        self.prompt_logprobs = prompt_logprobs
        self.prefilled_token_id = prefilled_token_id
        self.output_logprobs = output_logprobs
        self.cumulative_logprob = cumulative_logprob
        # self.speculate_token_ids = speculate_token_ids
        self.sampling_params = sampling_params
        self.index = index
        self.texts = texts
        self.finished = finished
    
    def __json__(self) -> Dict:
        prompt_logprobs = []
        print("self.prompt_logprobs ", self.prompt_logprobs)
        print("self.prompt_logprobs ", self.output_logprobs)
        for d in self.prompt_logprobs:
            if d == None:
                prompt_logprobs.append(d)
                continue
            serialized_d = {}
            for key, value in d.items():
                serialized_value = value.__json__()
                serialized_d[key] = serialized_value
            prompt_logprobs.append(serialized_d)
    
        output_logprobs = []
        for d in self.output_logprobs:
            if d == None:
                output_logprobs.append(d)
                continue
            serialized_d = {}
            for key, value in d.items():
                serialized_value = value.__json__()
                serialized_d[key] = serialized_value
            output_logprobs.append(serialized_d)
    
        return {
            "request_id": self.request_id,
            "opp_ranks": self.opp_ranks,
            "prompt_token_ids": self.prompt_token_ids,
            'prompt_logprobs': prompt_logprobs,
            "prefilled_token_id": self.prefilled_token_id,
            "output_logprobs": output_logprobs,
            "cumulative_logprob": self.cumulative_logprob,
            # "speculate_token_ids": self.speculate_token_ids,
            "sampling_params": self.sampling_params.__json__(),
            "index": self.index,
            "texts": self.texts,
            "finished": self.finished
        }
    