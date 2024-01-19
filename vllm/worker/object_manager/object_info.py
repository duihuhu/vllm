from typing import List
import pyarrow._plasma as plasma_object

class ObjectInfo:
  def __init__(self, request_id, seq_id, block_num, num_layers, device_id, ip_address) -> None:
    self.request_id = request_id
    self.seq_id = seq_id
    self.gpu_block_num = block_num
    self.num_layers = num_layers 
    self.device_id = device_id
    self.ip_address = ip_address
    self.object_ids = []
    self.rank = -1
  
  def allocate_objects_id(self, num_layers) -> List[List[plasma_object.ObjectID]]:
    object_ids = []
    key_object_ids = []  
    value_object_ids = [] 
    for i in range(num_layers):
      key_obj_id = plasma_object.ObjectID.from_random()
      key_object_ids.append(key_obj_id)
      value_obj_id = plasma_object.ObjectID.from_random()
      value_object_ids.append(value_obj_id)
    
    object_ids.append(key_object_ids)
    object_ids.append(value_object_ids)
    return object_ids