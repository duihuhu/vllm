import enum
from typing import List, Dict
from vllm.outputs import CompletionOutput, RequestOutput
import requests
from vllm.engine.plasma_client import plasma_client
import socket
import ctypes
from vllm.worker.object_manager.object_info import ObjectInfo
from vllm.sequence import SequenceData, SequenceGroupMetadata, SequenceOutputs, SequenceGroup
from vllm.config import ModelConfig, CacheConfig
from vllm.worker.cache_engine import CacheEngine
mine_ip = "127.0.0.1"
decode_info = {}
class NetStatus(enum.Enum):
    S_MEM = enum.auto()
    ROCE = enum.auto()
    SOCKET = enum.auto()

class MachineType(enum.Enum):
    mprefill = "mprefill"
    mdecode = "mdecode"

class KvTransfer:
  def __init__(self, machine_type, model_config, cache_config, parallel_config) -> None:
    if machine_type == "mprefill":
      self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      # 连接到服务器
      self.server_address = ('127.0.0.1', 12345)
    
    # self.rdma
    self.model_config = model_config
    self.cache_config = cache_config
    self.parallel_config = parallel_config
    pass
  
  def decode_machine_info(self):
    #mock one or two different machine 
    return mine_ip
  
  def judge_network_type(self):
    cur_ip = mine_ip
    dest_ip = self.decode_machine_info()
    if cur_ip == dest_ip:
      return NetStatus.S_MEM
    #elif qp status
    #  return NetStatus.ROCE
    return NetStatus.SOCKET

  def send(self, prefilled, prefill_blocks_to_object_swap_out):
    # net_type = self.judge_network_type()
    net_type = NetStatus.SOCKET
    if net_type == NetStatus.SOCKET:
        prefilled = self.send_in_socket(prefilled, prefill_blocks_to_object_swap_out)
    elif net_type == NetStatus.ROCE:
        # Handle ROCE case
        self.send_in_roce(prefill_blocks_to_object_swap_out)
        pass
    elif net_type == NetStatus.S_MEM:
        # Handle Shared Memory case
        prefilled = self.send_in_smem(prefilled)
        return prefilled
    return 
  
  def send_in_socket(self, prefilled, prefill_blocks_to_object_swap_out: Dict[int, List[ObjectInfo]]):
    self.client_socket.connect(self.server_address)
    #todo get from c
    kv_bytes = self._get_kv_size()
    obj_ids, obj_addr = self.get_kv_object_address(prefill_blocks_to_object_swap_out)
    # print("key value addr ", obj_ids, obj_addr, kv_bytes)
    self.send_to_mdecode(obj_ids, obj_addr, kv_bytes)
    self.client_socket.close()
    return prefilled
  
  def get_data_at_address(self, start_address, length):
      # 创建一个缓冲区，这里假设你要读取的是 bytes 数据
      buffer = ctypes.create_string_buffer(length)
      # 将指定地址的数据复制到缓冲区
      ctypes.memmove(buffer, start_address, length)
      # 获取缓冲区中的数据
      data = bytes(buffer)
      return data
    
  def send_to_mdecode(self, obj_ids, obj_addr, kv_bytes):
    obj_count = len(obj_ids)
    self.client_socket.sendall(obj_count.to_bytes(8, byteorder='big'))
    
    for obj, k_addr in zip(obj_ids, obj_addr):
      obj_str = obj.binary().hex()
      print("obj str ", obj_str)
      obj_bytes = obj_str.encode('utf-8')
      self.client_socket.sendall(len(obj_bytes).to_bytes(8, byteorder='big'))
      self.client_socket.sendall(obj_bytes)
      #send buffer size
      self.client_socket.sendall(kv_bytes.to_bytes(4, byteorder='big'))
      
      data = self.get_data_at_address(obj_addr, kv_bytes)
      # print("k_addr ", k_addr, type(k_addr), k_addr.to_bytes(byteorder='big'))
      # buffer = ctypes.create_string_buffer(kv_bytes)
      # mv = memoryview(buffer)
      # # mv[:] = k_addr.to_bytes(byteorder='big')
      # mv[:] = k_addr.to_bytes(kv_bytes, byteorder='big')
      # # 发送实际数据
      self.client_socket.sendall(data)
    return
  
  def get_kv_object_address(self, prefill_blocks_to_object_swap_out):
    rank = 0
    # print("_swap_in_prefilled_to_plasma rank ", rank, rank % self.parallel_config.tensor_parallel_size)
    # key_object_address = []
    # value_object_address = []
    object_ids = []
    object_address = []
    for key, obj_info in prefill_blocks_to_object_swap_out.items():
        for kobj, vobj in obj_info.items():
          for obj in vobj:
            for ids in obj.object_ids:
            # print("object ids " , obj.object_ids)
              object_ids.extend(ids)
    
    obj_plasma = plasma_client.get_buffers(object_ids)
    for obj in obj_plasma:
      object_address.append(obj.address)
    print("object_ids ", object_ids)
    print("object_address ", object_address)
        # print(key, "obj_info ", obj_info)
        # key_obj_info = (obj_info[rank].object_ids)[0]
        # kv_bytes = key_obj_info.kv_size
        # value_obj_info = (obj_info[rank].object_ids)[1]
        # key_obj_buf = plasma_client.get_buffers(key_obj_info)
        # value_obj_buf = plasma_client.get_buffers(value_obj_info)
        # key_obj_addr = []
        # value_obj_addr = []
        # for k_addr, v_addr in zip(key_obj_buf, value_obj_buf):
        #     key_obj_addr.append(k_addr.address)
        #     value_obj_addr.append(v_addr.address)
        # key_object_address.append(key_obj_addr)
        # value_object_address.append(value_obj_addr)
        
    return object_ids, object_address
  
  def recv_in_sockect():
    return
  
  def send_in_smem(self, prefilled) -> None:
    self.output_logprobs: List[Dict[int, float]] = []
    self.output_tokens: List[str] = []
    self.output_text = ""
    request_ids = []
    seq_ids = []
    prefilled_token_ids = []
    prefilled_texts = []
    cumulative_logprobs = []
    output_lens = []
    while prefilled:
        seq_group = prefilled.pop(0)
        request_outputs = RequestOutput.from_seq_group(seq_group)
        # print("request_output ", request_outputs)
        request_ids.append(request_outputs.request_id)
        output_lens.append(seq_group.sampling_params.max_tokens)
        # print("output_lens ", seq_group.sampling_params.max_tokens)
        seq_ids_pre_req = []
        prefilled_token_ids_in_req = []
        prefilled_text_in_req = []
        cumulative_logprob_in_req = []
        for output in request_outputs.outputs:
            seq_ids_pre_req.append(output.seq_id)
            prefilled_token_ids_in_req.append(output.token_ids)
            prefilled_text_in_req.append(output.text)
            cumulative_logprob_in_req.append(output.cumulative_logprob)
        seq_ids.append(seq_ids_pre_req)
        prefilled_token_ids.append(prefilled_token_ids_in_req)
        prefilled_texts.append(prefilled_text_in_req)
        cumulative_logprobs.append(cumulative_logprob_in_req)

    
    host='127.0.0.1'
    port = '9000'
    api_url = f"http://{host}:{port}/prefilled"
    headers = {"User-Agent": "Test Client"}
    pload = {
        "request_ids": request_ids,
        "output_lens": output_lens,
        "seq_ids": seq_ids,
        "prefilled_token_ids": prefilled_token_ids,
        "prefilled_texts": prefilled_texts,
        "cumulative_logprobs": cumulative_logprobs
    }
    response = requests.post(api_url, headers=headers, json=pload)
    # self.scheduler.watch_cpu_kv_cache()
    return prefilled
  
  #todo
  def send_in_roce(self, prefill_blocks_to_object_swap_out):
    return
  #todo
  def recv_in_roce(self, prefill_blocks_to_object_swap_out):
    return
  
  def _get_kv_size(self,):
      return 12288
      # cache_block_size = CacheEngine.get_cache_block_size(self.cache_config.block_size, self.model_config, self.parallel_config)
      # return cache_block_size