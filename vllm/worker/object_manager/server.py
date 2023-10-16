import zerorpc
from vllm.worker.object_manager.service import RPCService

class ObjectServer:
  def listen(self) -> None:
    self.server_ = zerorpc.Server(RPCService())
    self.server_.bind("tcp://0.0.0.0:4242")
    self.server_.run()

if __name__ == "__main__":
  server = ObjectServer()
  server.listen()


