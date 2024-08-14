import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
import argparse

def run_memory_efficient_attention_forward(
        num_tokens: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        seed: int,
        device: str,
        num_iters: int) -> None:
    
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.set_default_device(device)

    scale = float(1.0 / (head_size ** 0.5))

    qkv = torch.empty(num_tokens,
                      num_query_heads + 2 * num_kv_heads,
                      head_size,
                      dtype = dtype,
                      device = device).uniform_(-1e-3, 1e-3)
    query, key, value = qkv.split([num_query_heads, num_kv_heads, num_kv_heads], dim = 1)

    attn_bias = BlockDiagonalCausalMask.from_seqlens([num_tokens])

    for _ in range(num_iters):
        _ = xops.memory_efficient_attention_forward(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            attn_bias = attn_bias,
            p = 0.0,
            scale = scale)

def main(args: argparse.Namespace):
    run_memory_efficient_attention_forward(num_tokens = args.num_tokens,
                                           num_query_heads = args.num_query_heads,
                                           num_kv_heads = args.num_kv_heads,
                                           head_size = args.head_size,
                                           dtype = torch.float16,
                                           seed = args.seed,
                                           device = args.device,
                                           num_iters = args.num_iters)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'profile num. of thread blocks for a certain op')
    parser.add_argument('--num-tokens', type = int, default = 2048)
    parser.add_argument('--num-query-heads', type = int, default = 40)
    parser.add_argument('--num-kv-heads', type = int, default = 40)
    parser.add_argument('--head-size', type = int, default = 128)
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--device', type = str, default = 'cuda')
    parser.add_argument('--num-iters', type = int, default = 100)

    args = parser.parse_args()
    main(args)