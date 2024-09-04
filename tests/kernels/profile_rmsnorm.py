import torch
from vllm.model_executor.layers.layernorm import RMSNorm
import argparse

def run_rmsnorm(num_tokens: int,
                hidden_dim: int,
                dtype: torch.dtype,
                device: str,
                num_iters: int,
                seed: int) -> None:
    
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    rmsnorm = RMSNorm(hidden_size = 5120, eps = 1e-05)

    for _ in range(num_iters):
        hidden_states = torch.empty(num_tokens,
                                hidden_dim,
                                dtype = dtype,
                                device = device).uniform_(-1e-3, 1e-3)
        hidden_states = rmsnorm(hidden_states)
    
    for _ in range(num_iters):
        hidden_states = torch.empty(num_tokens,
                                hidden_dim,
                                dtype = dtype,
                                device = device).uniform_(-1e-3, 1e-3)
        residual = hidden_states
        hidden_states, residual = rmsnorm(hidden_states, residual)

def main(args: argparse.Namespace):
    run_rmsnorm(num_tokens = args.num_tokens,
                hidden_dim = args.hidden_dim,
                dtype = torch.float16,
                device = args.device,
                num_iters = args.num_iters,
                seed = args.seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'profile num. of thread blocks for a certain op')
    parser.add_argument('--num-tokens', type = int, default = 2048)
    parser.add_argument('--hidden-dim', type = int, default = 5120)
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--device', type = str, default = 'cuda')
    parser.add_argument('--num-iters', type = int, default = 100)

    args = parser.parse_args()
    main(args)


    

    

    
    