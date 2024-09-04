import torch
from vllm.model_executor.layers.linear import QKVParallelLinear
import argparse

def run_qkv_proj(hidde_size: int,
                 head_dim: int,
                 total_num_heads: int,
                 total_num_kv_heads: int) -> None:
    qkv_proj = QKVParallelLinear(hidden_size = hidde_size,
                                 head_size = head_dim,
                                 total_num_heads = total_num_heads,
                                 total_num_kv_heads = total_num_kv_heads,
                                 bias = False,
                                 linear_method = None)
    
    
