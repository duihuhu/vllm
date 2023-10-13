import zerorpc

class ObjectClient:
    def __init__(self) -> None:
      self.socket_client_ = zerorpc.Client()
      self.socket_client_.connect("tcp://127.0.0.1:4242")    
    