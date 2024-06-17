import argparse
from typing import Iterable, List, Optional, Tuple
import numpy as np 
import torch
import random


def get_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--request-rate", type=float, default=1)
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--input-len", type=int, default=1)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="ReAct", choices=["ShareGPT", "LooGLE", "ReAct"])
    parser.add_argument("--test-type", type=str, default="closed", choices=["open", "closed"])

    args = parser.parse_args()

    return args

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False