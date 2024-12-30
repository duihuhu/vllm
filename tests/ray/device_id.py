import ray
import time
import os 
import socket 
import warnings
def get_ip() -> str:
    host_ip = os.environ.get("HOST_IP")
    if host_ip:
        return host_ip

    # IP is not set, try to get it from the network interface

    # try ipv4
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    # try ipv6
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        # Google's public DNS server, see
        # https://developers.google.com/speed/public-dns/docs/using#addresses
        s.connect(("2001:4860:4860::8888", 80))  # Doesn't need to be reachable
        return s.getsockname()[0]
    except Exception:
        pass

    warnings.warn(
        "Failed to get the IP address, using 0.0.0.0 by default."
        "The value can be set by the environment variable HOST_IP.",
        stacklevel=2)
    return "0.0.0.0"

# 启动 Ray 集群
ray.init(address="auto")  # 自动连接到 Ray 集群

# 定义一个 Ray 远程函数，返回当前任务绑定的 GPU ID
@ray.remote(num_gpus=1)
def check_gpu():
    # 获取当前任务绑定的 GPU ID
    accelerator_ids = ray.get_runtime_context().get_accelerator_ids()["GPU"][0]
    # print(accelerator_ids, get_ip())
    ip = get_ip()
    time.sleep(3)
    return f"{accelerator_ids}-{ip}"

# 提交多个任务，验证不同任务的 GPU 绑定情况
def test_gpu_binding(num_tasks):
    futures = [check_gpu.remote() for _ in range(num_tasks)]
    results = ray.get(futures)
    
    for i, result in enumerate(results):
        print(f"Task {i} is using GPU(s): {result}")

# 假设我们有 2 台机器，每台机器有 8 张 GPU，总共 16 张 GPU
# 提交 16 个任务
test_gpu_binding(15)

# 关闭 Ray 集群
ray.shutdown()



