import enum
from typing import List, Dict
from vllm.outputs import CompletionOutput, RequestOutput
import requests
from vllm.engine.plasma_client import plasma_client
import socket
from ctypes import create_string_buffer

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
  def __init__(self, machine_type) -> None:
    if machine_type == "mprefill":
      self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      # 连接到服务器
      self.server_address = ('127.0.0.1', 12345)
    # self.rdma
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
  
  def send_in_socket(self, prefilled, prefill_blocks_to_object_swap_out):
    key_address, value_address, kv_bytes = self.get_kv_object_address(prefill_blocks_to_object_swap_out)
    print("key value addr ", key_address, value_address, kv_bytes)
    self.client_socket.connect(self.server_address)
    self.send_to_mdecode(key_address, value_address, kv_bytes)
    self.client_socket.close()
    return prefilled
  
  def send_to_mdecode(self, key_object_address, value_object_address, kv_bytes):
    for k_addr in key_object_address:
      self.client_socket.sendall(k_addr.to_bytes(8, byteorder='big'))
      self.client_socket.sendall(kv_bytes.to_bytes(4, byteorder='big'))
      buffer = create_string_buffer(kv_bytes)
      mv = memoryview(buffer)
      mv[:] = k_addr.to_bytes(kv_bytes, byteorder='big')
      # 发送实际数据
      self.client_socket.sendall(mv)
      
    for v_addr in value_object_address:
      self.client_socket.sendall(v_addr.to_bytes(8, byteorder='big'))
      self.client_socket.sendall(kv_bytes.to_bytes(4, byteorder='big'))
      buffer = create_string_buffer(kv_bytes)
      mv = memoryview(buffer)
      mv[:] = v_addr.to_bytes(kv_bytes, byteorder='big')
      # 发送实际数据
      self.client_socket.sendall(mv)
    return
  
  def get_kv_object_address(self, prefill_blocks_to_object_swap_out):
    rank = 0
    print("prefill_blocks_to_object_swap_out ", prefill_blocks_to_object_swap_out)
    block_size_in_bytes = prefill_blocks_to_object_swap_out[0][1].element_size() * prefill_blocks_to_object_swap_out[0][1][0].numel()
        # print("_swap_in_prefilled_to_plasma rank ", rank, rank % self.parallel_config.tensor_parallel_size)
    src_to_dst_copy = {}
    key_object_address = []
    value_object_address = []
    for key, obj_info in prefill_blocks_to_object_swap_out.items():
        src_to_dst_copy[key] = 0
        key_obj_info = (obj_info[rank].object_ids)[0]
        value_obj_info = (obj_info[rank].object_ids)[1]
        key_obj_buf = plasma_client.get_buffers(key_obj_info)
        value_obj_buf = plasma_client.get_buffers(value_obj_info)
        key_obj_addr = []
        value_obj_addr = []
        for k_addr, v_addr in zip(key_obj_buf, value_obj_buf):
            key_obj_addr.append(k_addr.address)
            value_obj_addr.append(v_addr.address)
        key_object_address.append(key_obj_addr)
        value_object_address.append(value_obj_addr)
        
    return key_object_address, value_object_address, block_size_in_bytes
  
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
  