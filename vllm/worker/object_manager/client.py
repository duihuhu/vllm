import zerorpc
import pickle
from vllm.worker.object_manager.object_info import ObjectInfo
class ObjectClient:
    def __init__(self) -> None:
      self.socket_client_ = zerorpc.Client()
      self.socket_client_.connect("tcp://127.0.0.1:4242")    


if __name__ == "__main__":
  client = ObjectClient()
  # mock prompt id , tensor is kv cache, object id is plasma primary key
  request_id = 1
  seq_id = 1
  gpu_block_nums = [234,567]
  num_layers = 12
  device_id = 1
  ip_address = 1
  obj = client.socket_client_.create_objects_id(request_id, seq_id, gpu_block_nums, num_layers, device_id, ip_address)
  objs = pickle.loads(obj)
  for key, value in objs.items():
    for object_id in value.object_ids:
      print(key, object_id.binary().hex())
