#run plasma server: ./plasma-store-server -m 4000000000 -s /tmp/plasma_store
import pyarrow._plasma as plasma_object
from vllm.worker.object_manager.object_info import ObjectId
import pickle
# class PlasmaClient:
#     def __init__(self, plasma_store_socket_name) -> None:
#         self.plasma_client_ = plasma_object.connect(plasma_store_socket_name)
    
#     def allocate_object_id(self):
#         obj_id = plasma_object.ObjectID.from_random()
#         return obj_id
    
    # def create(self, object_id, length):
    #     obj = self.plasma_client_.create(object_id, length)
    #     return obj
    
    # def seal(self, object_id):
    #     self.plasma_client_.seal(object_id)
  
    # def get_buffers(self, object_id):
    #     return self.plasma_client_.get_buffers([object_id])

class RPCService(object):
  def __init__(self) -> None:
    self.request_table_ = {}
    self.seq_table_ = {}
    
  def create_objects_id(self, request_id, seq_id, gpu_block_nums, num_layers, device_id, ip_address):
    block_object = {}
    for block_num in gpu_block_nums:
      objects_id = ObjectId(request_id, seq_id, block_num, num_layers, device_id, ip_address)
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
    # return block_object

