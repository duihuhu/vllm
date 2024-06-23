from vllm.radix_tree_ys.radix_tree_manager import RadixTreeManager
from vllm.sequence import Sequence
import numpy as np
import argparse
import math
from typing import List

BLOCK_SIZE = 16

def get_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-len", type=int, default=2048)
    parser.add_argument("--cache-ratio", type=float, default=0.5)

    args = parser.parse_args()

    return args

def generate_random_token_ids(
    input_len: int, 
) -> List[int]:
    return list(np.random.randint(0, 10000, input_len))

if __name__ == "__main__":

    manager = RadixTreeManager(BLOCK_SIZE)
    args = get_args()


    second_seq = Sequence(2, None, 
        generate_random_token_ids(args.input_len),
        BLOCK_SIZE
    )
    first_seq = Sequence(1, None, 
        second_seq.data.prompt_token_ids[:math.ceil(args.input_len * args.cache_ratio)],
        BLOCK_SIZE
    )
    
    import pdb; pdb.set_trace()

    manager.insert(first_seq, None)
    manager.match(second_seq)

