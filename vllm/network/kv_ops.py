import enum
class NetStatus(enum.Enum):
    S_MEM = enum.auto()
    ROCE = enum.auto()
    SOCKET = enum.auto()
    
class KvOps:
  def __init__(self) -> None:
    pass
  
  def judge_network_type(self, p_ip, d_ip):
    if p_ip == d_ip:
      return NetStatus.S_MEM
    #elif qp status
    #  return NetStatus.ROCE
    return NetStatus.SOCKET

  def send(self, p_ip, d_ip, blocks):
    net_type = self.judge_network_type(p_ip, d_ip)
    if net_type == NetStatus.SOCKET:
        self.send_in_socket(blocks)
    elif net_type == NetStatus.ROCE:
        # Handle ROCE case
        self.send_in_roce()
        pass
    elif net_type == NetStatus.S_MEM:
        # Handle Shared Memory case
        self.send_in_smem()
    return 
  def send_in_socket():
    return
  
  def recv_in_sockect():
    return 
  
  def send_in_smem():
    return
  
  def recv_in_smem():
    return
  
  def send_in_roce():
    return
  
  def recv_in_roce():
    return