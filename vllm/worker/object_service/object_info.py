from typing import List
import pyarrow._plasma as plasma_object

class ObjectId:
  def __init__(self, request_id, seq_id, block_num, num_layers, device_id, ip_address) -> None:
    self.request_id = request_id
    self.seq_id = seq_id
    self.gpu_block_num = block_num
    self.num_layers = num_layers 
    self.device_id = device_id
    self.ip_address = ip_address
    self.object_ids = []
  
  def allocate_objects_id(self, num_layers) -> List[plasma_object.ObjectID]:
    object_ids = []
    for i in range(num_layers):
      obj_id = plasma_object.ObjectID.from_random()
      object_ids.append(obj_id)
    return object_ids