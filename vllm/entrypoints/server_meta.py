from typing import List, Dict, Optional

class InferResults:
    def __init__(
        self,
        request_id,
        opp_ranks: Optional[List[int]] = None,
        prompt_token_ids: Optional[List[int]] = None,
        prompt_logprobs: Optional[List[int]] = None,
        prefilled_token_id: Optional[List[int]] = None,
        output_logprobs: Optional[List[int]] = None,
        cumulative_logprob: Optional[List[int]] = None,
        sampling_params: Optional[List[int]] = None,
        index: Optional[List[int]] = None,
        texts:  Optional[List[str]] = None,
        finished: Optional[bool] = False,
        ttft: Optional[int] = 0,
        jct: Optional[int] = 0,
        tbt: Optional[int] = 0,
        n: Optional[int] =  0,
        start_time: Optional[int] = 0,
        end_time: Optional[int] = 0,
        is_layer: Optional[bool] = False,
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
        self.ttft =ttft
        self.jct = jct
        self.tbt = tbt
        self.n = n
        self.start_time = start_time
        self.end_time = end_time
        self.is_layer = is_layer
    def __json__(self) -> Dict:
        prompt_logprobs = []
        if self.prompt_logprobs != None:
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
        if self.output_logprobs != None:
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
            "finished": self.finished,
            "ttft": self.ttft,
            "jct": self.jct,
            "tbt": self.tbt,
            "n": self.n,
            "start_time": self.start_time,
            "end_time": self.end_time,   
            "is_layer": self.is_layer, 
        }

class QueryLayerKvBlocks:
    def __init__(self, request_id , prompt_token_ids, sampling_params, global_ranks) -> None:
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.sampling_params = sampling_params
        self.global_ranks = global_ranks

    def __json__(self) -> Dict:
        return {
            "request_id": self.request_id,
            "prompt_token_ids": self.prompt_token_ids,
            "sampling_params": self.sampling_params.__json__(),
            "global_ranks": self.global_ranks
        }
class PrefilledMeta:
    def __init__(self, request_id, prefilled_token_ids,  output_logprobs) -> None:
        self.request_id = request_id
        self.prefilled_token_ids = prefilled_token_ids
        self.output_logprobs = output_logprobs
        
    def __json__(self) -> Dict:
        output_logprobs = []
        if self.output_logprobs != None:
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
            "prefilled_token_ids": self.prefilled_token_ids,
            "output_logprobs": output_logprobs,
        }