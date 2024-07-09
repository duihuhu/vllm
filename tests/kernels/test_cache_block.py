import os
import torch
import time
import random
import math
from vllm._C import cache_ops, ops

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

num_layers = 40
num_blocks = 300
num_kv_heads = 20
head_size = 128
block_size = 16
#x = 16 // torch.tensor([], dtype=torch.float16).element_size()

'''key_agg_blocks = []
for _ in range(num_blocks):
    key_block_tensor = torch.empty(size = (2, num_layers, num_kv_heads, head_size // x, block_size, x), 
                                 dtype=torch.float16, 
                                 device='cuda').uniform_(-1e-3, 1e-3)
    key_agg_blocks.append(key_block_tensor)
value_agg_blocks = []
for _ in range(num_blocks):
    value_block_tensor = torch.empty(size = (2, num_layers, num_kv_heads, head_size, block_size), 
                                 dtype=torch.float16, 
                                 device='cuda').uniform_(-1e-3, 1e-3)
    value_agg_blocks.append(value_block_tensor)

key_blocks_addresses = ops.tensor_for_caches_addresses(key_agg_blocks)
value_blocks_addresses = ops.tensor_for_caches_addresses(value_agg_blocks)'''

gpu_cache = []
for _ in range(num_layers):
    gpu_block_tensor = torch.zeros(size = (2, num_blocks, num_kv_heads * head_size * block_size), 
                                 dtype = torch.float16, 
                                 device = 'cuda:1')
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

print("----------Warm Up----------")
for _ in range(10):
     cache_ops.swap_blocks(cpu_cache[0][0], gpu_cache[0][0], unique_dicts[0])
print("---------End---------")

print("----------Sleep----------")
time.sleep(5)
print("----------End----------")

print("-----------Start----------")
outputs = []
for i in range(3):
    for j in range(10):
        slots = []
        k = i * 10 + j
        unique_dict = unique_dicts[k]
        t = 0
        print(f"Length {lengths[i]} Ratio {ratios[j]}")
        print(f"K {k}")
        print(f"Len Map {len(unique_dict.items())}")
        print(unique_dict)
        for _ in range(3):
            st = time.time()
            for layer in range(num_layers):
                cache_ops.swap_blocks(cpu_cache[layer][0], gpu_cache[layer][0], unique_dict)
                cache_ops.swap_blocks(cpu_cache[layer][1], gpu_cache[layer][1], unique_dict)
            ed = time.time()
            t = t + ed - st
        slots.append(t / 3)
    outputs.append(slots)
print("----------End-----------")

print("---------Outputs---------")
print(outputs)

'''block_mapping = {}
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
    print("Error in Swap")'''

'''block_mapping2 = {}
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