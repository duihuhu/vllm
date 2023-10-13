import zerorpc
from vllm.worker.object_info import ObjectId
import pyarrow._plasma as plasma_object

class ObjectClient:
    def __init__(self) -> None:
      self.socket_client_ = zerorpc.Client()
      self.socket_client_.connect("tcp://127.0.0.1:4242")    
    