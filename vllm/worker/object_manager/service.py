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
      objects_id = ObjectInfo(request_id, seq_id, block_num, num_layers, device_id, ip_address)
      objects_id.object_ids = objects_id.allocate_objects_id(num_layers)
      for object_id in objects_id.object_ids:
        print(block_num, object_id.binary().hex())
      if seq_id in self.seq_table_:
        self.seq_table_[seq_id].append(objects_id)
      else:
        self.seq_table_[seq_id] = [objects_id]
      block_object[block_num] = objects_id
    ser_block_object = pickle.dumps(block_object)
    return ser_block_object
