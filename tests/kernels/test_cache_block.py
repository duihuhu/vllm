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

block_mapping = {0:2}

block_size_in_bytes = key_agg_blocks[0].numel() * key_agg_blocks[0].element_size()

#copy from keys to values -> simplize the test
cache_ops.swap_blocks_agg(key_blocks_addresses, value_blocks_addresses, block_mapping, block_size_in_bytes)

is_close = torch.allclose(key_agg_blocks[0], value_agg_blocks[2], atol=1e-3, rtol=1e-5)
if is_close:
    print("Pass")
else:
    print("Error")

block_mapping2 = {0:[3,4]}
cache_ops.copy_blocks_agg(key_blocks_addresses, value_blocks_addresses, key_agg_blocks[0][0,0,:,0,:], 
                          block_mapping2, num_layers, key_agg_blocks[0][0].numel())

is_close2 = torch.allclose(key_agg_blocks[0], value_agg_blocks[3], atol=1e-3, rtol=1e-5)
is_close3 = torch.allclose(key_agg_blocks[0], value_agg_blocks[4], atol=1e-3, rtol=1e-5)
if is_close2 and is_close3:
    print("Pass2")
else:
    print("Error2")


key = torch.empty(size = (16, 10, 128), dtype=torch.float16, device='cuda').uniform_(-1e-3, 1e-3)
value = torch.empty(size = (16, 10, 128), dtype=torch.float16, device='cuda').uniform_(-1e-3, 1e-3)
slots = [i for i in range(16)]
cache_ops.reshape_and_cache_agg(key, value, key_blocks_addresses, value_blocks_addresses, slots, "auto", 
                                block_size, x, 0)
