#run plasma server: ./plasma-store-server -m 4000000000 -s /tmp/plasma_store
import pyarrow._plasma as plasma_object
from vllm.worker.object_manager.object_info import ObjectInfo
import pickle
class RPCService(object):
  def __init__(self) -> None:
    self.request_table_ = {}
    self.seq_table_ = {}
    
  def create_objects_id(self, request_id, seq_id, gpu_block_nums, num_layers, device_id, ip_address):
    block_object = {}
    for block_num in gpu_block_nums:
      object_info = ObjectInfo(request_id, seq_id, block_num, num_layers, device_id, ip_address)
      object_info.object_ids = object_info.allocate_objects_id(num_layers)
      for object_id in object_info.object_ids:
        print(block_num, object_id.binary().hex())
      if seq_id in self.seq_table_:
        self.seq_table_[seq_id].append(object_info)
      else:
        self.seq_table_[seq_id] = [object_info]
      block_object[block_num] = object_info
    ser_block_object = pickle.dumps(block_object)
    print("ser_block_object ", ser_block_object)
    return ser_block_object
