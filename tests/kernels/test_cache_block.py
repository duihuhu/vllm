import os
import torch
import time
import random
import math
from typing import Dict, List, Tuple, Optional
from vllm._C import cache_ops

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def swap_blocks_vllm(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int]) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)

        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        cache_ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

def swap_blocks_agg(
        src_kv: torch.Tensor,
        dst_kv: torch.Tensor,
        block_size_in_bytes: int) -> None:
        cache_ops.swap_agg_block(src_kv, dst_kv, block_size_in_bytes)

def swap_vllm(cpu_cache: List[torch.Tensor],
              gpu_cache: List[torch.Tensor],
              num_layers, 
              src_to_dst: Dict[int, int]
              ) -> None:
        for i in range(num_layers):
            swap_blocks_vllm(cpu_cache[i], gpu_cache[i], src_to_dst)

def swap_agg(cpu_cache: List[torch.Tensor], 
             gpu_cache: List[torch.Tensor],
             src_to_dst: Dict[int, int]) -> None:
        block_size_in_bytes = cpu_cache[0].element_size() * cpu_cache[0].numel()
        for src, dst in src_to_dst.items():
            swap_blocks_agg(cpu_cache[src], gpu_cache[dst], block_size_in_bytes)

def get_tensors(num_layers: int,
                num_blocks: int,
                num_kv_heads: int,
                tp: int,
                head_size: int,
                block_size: int,
                device: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
     agg_blocks = []
     for _ in range(num_blocks):
          agg_block_tensor = torch.empty(size = (2, num_layers, num_kv_heads//tp * head_size * block_size),
                                         dtype = torch.float16, # use fp16 by default
                                         device = device).uniform_(-1e-3, 1e-3)
          agg_blocks.append(agg_block_tensor)
     vllm_tensors = []
     for _ in range(num_layers):
          vllm_layer_tensor = torch.empty(size = (2, num_blocks, num_kv_heads//tp * head_size * block_size),
                                          dtype = torch.float16,
                                          device = device).uniform_(-1e-3, 1e-3)
          vllm_tensors.append(vllm_layer_tensor)
     return (agg_blocks, vllm_tensors)

def get_mappings(seed: int,
                 num_blocks: int,
                 block_size: int) -> List[Dict[int, int]]:
    #random.seed(seed)
    all_keys = list(range(num_blocks))
    all_values = list(range(num_blocks))
    #random.shuffle(all_keys)
    #random.shuffle(all_values)
    unique_dicts = []
    lengths = [1024, 2048, 4096] # only test the useful lengths
    ratios = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # tests all ratios
    for length in lengths:
        for ratio in ratios:
            blocks_num = math.ceil(math.ceil(length * (ratio / 100)) / block_size)
            unique_dict = {}
            for i in range(blocks_num):
                key = all_keys[i]
                value = all_values[i]
                unique_dict[key] = value
            unique_dicts.append(unique_dict)
    return unique_dicts

def warm_up(iters: int,
            src_kv: torch.Tensor,
            dst_kv: torch.Tensor,
            src_kv_cache: torch.Tensor,
            dst_kv_cache: torch.Tensor,
            block_size_in_bytes: int,
            src_to_dst: Dict[int, int]) -> None:
     for _ in range(iters):
          cache_ops.swap_agg_block(src_kv, dst_kv, block_size_in_bytes)
     for _ in range(iters):
          cache_ops.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

def test_swap(unique_dicts: List[Dict[int, int]],
              agg: bool,
              num_layers: int,
              agg_cpu_cache: Optional[List[torch.Tensor]] = None,
              agg_gpu_cache: Optional[List[torch.Tensor]] = None,
              vllm_cpu_cache: Optional[List[torch.Tensor]] = None,
              vllm_gpu_cache: Optional[List[torch.Tensor]] = None) -> None:
    num_lengths = 3
    num_ratios = 10
    num_iters = 10
    outputs = []
    for i in range(num_lengths):
        slots = []
        for j in range(num_ratios):
            k = i * num_ratios + j
            unique_dict = unique_dicts[k]
            temp = []
            for _ in range(num_iters):
                st = time.time()
                if agg:
                     swap_agg(agg_cpu_cache, agg_gpu_cache, unique_dict)
                else:
                     swap_vllm(vllm_cpu_cache, vllm_gpu_cache, num_layers, unique_dict)
                ed = time.time()               
                temp.append(ed - st)
            slots.append(sum(temp) / len(temp))
        outputs.append(slots)
    print(outputs)

def test() -> None:
     num_layers = 40
     num_blocks = 300
     num_kv_heads = 40
     tp = 2
     head_size = 128
     block_size = 16
     seed = 42
     warm_ites = 10

     agg_cpu_cache, vllm_cpu_cache = get_tensors(num_layers, num_blocks, num_kv_heads, tp, head_size, block_size, 'cpu')
     agg_gpu_cache, vllm_gpu_cache = get_tensors(num_layers, num_blocks, num_kv_heads, tp, head_size, block_size, 'cuda')

     unique_dicts = get_mappings(seed, num_blocks, block_size)

     block_size_in_bytes = agg_cpu_cache[0].numel() * agg_cpu_cache[0].element_size()

     print("----------Warm Up----------")
     warm_up(warm_ites, agg_cpu_cache[0], agg_gpu_cache[0], vllm_cpu_cache[0], vllm_gpu_cache[0], block_size_in_bytes, 
             unique_dicts[0])
     print("----------End----------")

     print("-----------Test Agg----------")
     test_swap(unique_dicts = unique_dicts, agg = True, num_layers = num_layers, agg_cpu_cache = agg_cpu_cache, 
               agg_gpu_cache = agg_gpu_cache)
     print("-----------End----------")

     print("------------Test vllm----------")
     test_swap(unique_dicts = unique_dicts, agg = False, num_layers = num_layers, vllm_cpu_cache = vllm_cpu_cache, 
               vllm_gpu_cache = vllm_gpu_cache)
     print("-----------End----------")

test()

'''num_layers = 40
num_blocks = 300
num_kv_heads = 20
head_size = 128
block_size = 16
x = 16 // torch.tensor([], dtype=torch.float16).element_size()

gpu_agg_blocks = []
for _ in range(num_blocks):
    gpu_block_tensor = torch.zeros(size = (2, num_layers, num_kv_heads * head_size * block_size), 
                                 dtype=torch.float16, 
                                 device='cuda')
    gpu_agg_blocks.append(gpu_block_tensor)
cpu_agg_blocks = []
for _ in range(num_blocks):
    cpu_block_tensor = torch.zeros(size = (2, num_layers, num_kv_heads * head_size * block_size), 
                                 dtype=torch.float16, 
                                 device='cpu')
    cpu_agg_blocks.append(cpu_block_tensor)

key_blocks_addresses = ops.tensor_for_caches_addresses(cpu_agg_blocks)
value_blocks_addresses = ops.tensor_for_caches_addresses(gpu_agg_blocks)

gpu_cache = []
for _ in range(num_layers):
    gpu_block_tensor = torch.zeros(size = (2, num_blocks, num_kv_heads * head_size * block_size), 
                                 dtype = torch.float16, 
                                 device = 'cuda')
    gpu_cache.append(gpu_block_tensor)
cpu_cache = []
for _ in range(num_layers):
    cpu_block_tensor = torch.zeros(size = (2, num_blocks, num_kv_heads * head_size * block_size), 
                                 dtype = torch.float16, 
                                 device = 'cpu')
    cpu_cache.append(cpu_block_tensor)

random.seed(66)
all_keys = list(range(num_blocks))
all_values = list(range(num_blocks))
random.shuffle(all_keys)
random.shuffle(all_values)
unique_dicts = []
lengths = [1024, 2048, 4096]
ratios = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for length in lengths:
    for ratio in ratios:
        blocks_num = math.ceil(math.ceil(length * (ratio / 100)) / block_size)
        unique_dict = {}
        for i in range(blocks_num):
            key = all_keys[i]
            value = all_values[i]
            unique_dict[key] = value
        unique_dicts.append(unique_dict)

block_size_in_bytes = gpu_agg_blocks[0].numel() * gpu_agg_blocks[0].element_size()

print("----------Warm Up----------")
t = 0
for _ in range(10):
    s = time.time()
    cache_ops.swap_agg_block(cpu_agg_blocks[0], gpu_agg_blocks[0], block_size_in_bytes)
    e = time.time()
    t = t + (e - s)
print(t / 10)
    #cache_ops.swap_blocks(cpu_cache[0][0], gpu_cache[0][0], unique_dicts[0])
print("---------End---------")

print("----------Sleep----------")
time.sleep(5)
print("----------End----------")

print("-----------Start----------")
outputs = []
for i in range(3):
    slots = []
    for j in range(10):
        k = i * 10 + j
        unique_dict = unique_dicts[k]
        temp = []
        print(f"Length {lengths[i]} Ratio {ratios[j]}")
        print(f"K {k}")
        print(f"Len Map {len(unique_dict.items())}")
        print(unique_dict)
        for _ in range(10):
            st = time.time()
            for src, dst in unique_dict.items():
                cache_ops.swap_agg_block(cpu_agg_blocks[src], gpu_agg_blocks[dst], block_size_in_bytes)
                #cache_ops.swap_blocks(cpu_cache[layer][0], gpu_cache[layer][0], unique_dict)
                #cache_ops.swap_blocks(cpu_cache[layer][1], gpu_cache[layer][1], unique_dict)
            ed = time.time()
            #print(ed - st)
            temp.append(ed - st)
        #t = t + (ed - st)
        slots.append(sum(temp) / len(temp))
    outputs.append(slots)
print("----------End-----------")

print("---------Outputs---------")
print(outputs)

block_mapping = {}
block_mapping[2] = 4

block_size_in_bytes = key_agg_blocks[0].numel() * key_agg_blocks[0].element_size()

#copy from keys to values -> simplize the test
a = []
for _ in range(10):
    t1 = time.time()
    #cache_ops.swap_blocks_agg(key_blocks_addresses, key_blocks_addresses, block_mapping, block_size_in_bytes)
    cache_ops.swap_agg_block(key_agg_blocks[2], key_agg_blocks[4], block_size_in_bytes)
    t2 = time.time()
    a.append(t2-t1)

b = []
for _ in range(10):
    t3 = time.time()
    for _ in range(num_layers):
        cache_ops.swap_blocks(key_cache[0], key_cache[0], block_mapping)
    t4 = time.time()
    b.append(2*(t4 - t3))

print(f"swap_blocks_agg costs {sum(a) / len(a)}, swap_blocks costs {sum(b) / len(b)}")

is_close = torch.allclose(key_agg_blocks[2], key_agg_blocks[4], atol=1e-3, rtol=1e-5)
if is_close:
    print("Pass for Swap")
else:
    print("Error in Swap")

block_mapping2 = {}
block_mapping2[1] = [3,4]

t5 = time.time()
cache_ops.copy_blocks_agg(key_blocks_addresses, key_blocks_addresses, value_agg_blocks[0][0,0,:,0], 
                          block_mapping2, num_layers, key_agg_blocks[0][0].numel())
t6 = time.time()

t7 = time.time()
cache_ops.copy_blocks(key_cache, key_cache, block_mapping2)
t8 = time.time()

print(f"copy_blocks_agg costs {t6-t5}, copy_blocks costs {t8-t7}")

is_close2 = torch.allclose(key_agg_blocks[1], key_agg_blocks[3], atol=1e-3, rtol=1e-5)
is_close3 = torch.allclose(key_agg_blocks[1], key_agg_blocks[4], atol=1e-3, rtol=1e-5)
if is_close2 and is_close3:
    print("Pass for Copy")
else:
    print("Error in Copy")


key = torch.zeros(size = (16, 10, 128), dtype=torch.float16, device='cuda')
value = torch.zeros(size = (16, 10, 128), dtype=torch.float16, device='cuda')
slots = [i for i in range(16)]
slots2 = torch.tensor(slots, dtype=torch.long).to('cuda')

t9 = time.time()
cache_ops.reshape_and_cache_agg(key, value, key_blocks_addresses, value_blocks_addresses, slots2, "auto", 
                                block_size, x, 0)
t10 = time.time()

t11 = time.time()
cache_ops.reshape_and_cache(key, value, key_cache[1], value_cache[1], slots2, "auto")
t12 = time.time()

print(f"reshape_and_cache_agg costs {t10-t9}, reshape_and_cache costs {t12-t11}")

all_zero = torch.all(key_agg_blocks[0][0] == 0)
if all_zero:
    print("Pass for Store")
else:
    print("Error in Store")'''