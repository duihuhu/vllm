from enum import Enum

#record session id , prompt, output, total text len, kv size, prefill machine, decode machine
class ReqCacheInfo:
    def __init__(self, request_id) -> None:
        self.request_id = request_id
        self.token = []
        self.prefilled_prompts = []
        self.unprefilled_prompts= []
        self.outputs = []
        self.eprefill_host = None
        self.eprefill_port = None
        self.edecode_host = None
        self.edecode_port = None
        self.epd_host = None
        self.epd_port = None

    def add_prefilled_prompt(self, prompt) -> None:
        self.prefilled_prompts.append(prompt)
        
    def add_unprefilled_prompt(self, prompt) -> None:
        self.unprefilled_prompts.append(prompt) 

    def add_token(self, tokens) ->None:
        self.token.extend(tokens)
     
    def add_output(self, output) -> None:
        self.outputs.append(output)

class InstanceInfo:
    def __init__(self, 
                 host, 
                 service_port, 
                 num_unfinished_reqs, 
                 used_gpu_blocks, 
                 used_cpu_blocks, 
                 remained_gpu_blocks,
                 remained_cpu_blocks,
                 engine_type,
                 timestamp,
                 global_ranks=None) -> None:
        self.host = host
        self.service_port = service_port
        self.num_unfinished_reqs = num_unfinished_reqs
        self.used_gpu_blocks = used_gpu_blocks
        self.used_cpu_blocks = used_cpu_blocks 
        self.remained_gpu_blocks = remained_gpu_blocks
        self.remained_cpu_blocks = remained_cpu_blocks
        self.engine_type = engine_type
        self.timestamp = timestamp
        self.global_ranks = global_ranks
        
class TransDataType(Enum):
    PART = "incr"
    FULL = "full"
    
class PrefixReqInfo:
  def __init__(self, request_id, prefix_type, matched_len, edecode_host, edecode_port, data_type):
      self.request_id = request_id
      self.prefix_type = prefix_type
      self.matched_len = matched_len
      self.edecode_host = edecode_host
      self.edecode_port = edecode_port
      self.data_type = data_type

class DistPolicy(Enum):
    RANDOM = "random"
    RR = "rr"
    PREFIX_CACHE = "prefix"
    LEAST_LOAD = "least"
    