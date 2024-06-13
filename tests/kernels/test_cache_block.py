import os
import torch
import time
from vllm._C import cache_ops, ops

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

num_layers = 40
num_blocks = 10
num_kv_heads = 10
head_size = 128
block_size = 16
x = 16 // torch.tensor([], dtype=torch.float16).element_size()

key_agg_blocks = []
for _ in range(num_blocks):
    key_block_tensor = torch.empty(size = (num_layers, num_kv_heads, head_size // x, block_size, x), 
                                 dtype=torch.float16, 
                                 device='cuda').uniform_(-1e-3, 1e-3)
    key_agg_blocks.append(key_block_tensor)
value_agg_blocks = []
for _ in range(num_blocks):
    value_block_tensor = torch.empty(size = (num_layers, num_kv_heads, head_size, block_size), 
                                 dtype=torch.float16, 
                                 device='cuda').uniform_(-1e-3, 1e-3)
    value_agg_blocks.append(value_block_tensor)

key_blocks_addresses = ops.tensor_for_caches_addresses(key_agg_blocks)
value_blocks_addresses = ops.tensor_for_caches_addresses(value_agg_blocks)

key_cache = []
for _ in range(num_layers):
    key_block_tensor2 = torch.empty(size = (num_blocks, num_kv_heads, head_size // x, block_size, x), 
                                 dtype=torch.float16, 
                                 device='cuda').uniform_(-1e-3, 1e-3)
    key_cache.append(key_block_tensor2)

value_cache = []
for _ in range(num_layers):
    value_block_tensor2 = torch.empty(size = (num_blocks, num_kv_heads, head_size // x, block_size, x), 
                                 dtype=torch.float16, 
                                 device='cuda').uniform_(-1e-3, 1e-3)
    value_cache.append(value_block_tensor2)

block_mapping = {}
block_mapping[0] = 2

block_size_in_bytes = key_agg_blocks[0].numel() * key_agg_blocks[0].element_size()

#copy from keys to values -> simplize the test
t1 = time.time()
cache_ops.swap_blocks_agg(key_blocks_addresses, key_blocks_addresses, block_mapping, block_size_in_bytes)
t2 = time.time()

t3 = time.time()
cache_ops.swap_blocks(key_cache[0], key_cache[0], block_mapping)
t4 = time.time()

print(f"swap_blocks_agg costs {t2-t1}, swap_blocks costs {t4-t3}")

is_close = torch.allclose(key_agg_blocks[0], key_agg_blocks[2], atol=1e-3, rtol=1e-5)
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
slots = torch.Tensor(slots, device='cuda')

t9 = time.time()
cache_ops.reshape_and_cache_agg(key, value, key_blocks_addresses, value_blocks_addresses, slots, "auto", 
                                block_size, x, 0)
t10 = time.time()

t11 = time.time()
cache_ops.reshape_and_cache(key, value, key_cache[1], value_cache[1], slots, "auto")
t12 = time.time()

print(f"reshape_and_cache_agg costs {t10-t9}, reshape_and_cache costs {t12-t11}")

all_zero = torch.all(key_agg_blocks[0][0] == 0)
if all_zero:
    print("Pass for Store")
else:
    print("Error in Store")