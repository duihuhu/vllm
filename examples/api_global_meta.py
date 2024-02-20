from enum import Enum
class MachineType(Enum):
    MPREFILL = "mp"
    MDECODE = "md"
    MPD = "mpd"
    
#record session id , prompt, output, total text len, kv size, prefill machine, decode machine
class ReqCacheInfo:
    def __init__(self, session_id, request_id) -> None:
        self.session_id = session_id
        self.request_id = request_id
        # self.src_url = None
        self.token = []
        self.prefilled_prompts = []
        self.unprefilled_prompts= []
        self.outputs = []
        # self.text_size = None
        # self.kv_size = None
        self.mprefill_host = None
        self.mprefill_port = None
        self.mdecode_host = None
        self.mdecode_port = None
        # self.pd_type = None
        # pass

    def add_prefilled_prompt(self, prompt) -> None:
        self.prefilled_prompts.append(prompt)
        return 

    def add_unprefilled_prompt(self, prompt) -> None:
        self.unprefilled_prompts.append(prompt)
        return  

    def add_token(self, tokens) ->None:
        self.token.extend(tokens)
        return
     
    def add_output(self, output) -> None:
        self.outputs.append(output)
        return

class InstanceInfo:
    def __init__(self, host, service_port, unfinished_reqs, used_gpu_blocks, used_cpu_blocks, remained_gpu_blocks, remained_cpu_blocks, machine_type, timestamp) -> None:
        self.host = host
        self.service_port = service_port
        self.unfinished_reqs = unfinished_reqs
        self.used_gpu_blocks = used_gpu_blocks
        self.used_cpu_blocks = used_cpu_blocks 
        self.remained_gpu_blocks = remained_gpu_blocks
        self.remained_cpu_blocks = remained_cpu_blocks
        self.machine_type = machine_type
        self.timestamp = timestamp
        
class TransDataType(Enum):
    PART = "incr"
    FULL = "full"
    
class PrefixReqInfo:
  def __init__(self, request_id, type, matched_len, mdecode_host, mdecode_port, data_type):
      self.request_id = request_id
      self.type = type
      self.matched_len = matched_len
      self.mdecode_host = mdecode_host
      self.mdecode_port = mdecode_port
      self.data_type = data_type
