from vllm.radix_tree_ys.radix_cache import RadixCache, TreeNodeValue, TreeNode, kvCacheProgressStatus
import numpy as np
import argparse
import math
from typing import List, Tuple
from vllm.block import PhysicalTokenBlock
import time 


def get_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_blocks", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--cache-ratio", type=float, default=0.5)
    parser.add_argument("--repeat-times", type=int, default=10)
    # parser.add_argument("--test-type", type=str, default='match', choices=['insert', 'match'])

    args = parser.parse_args()

    return args

def generate_random_token_ids(
    num_token_ids: int, 
) -> Tuple[int]:
    return tuple(np.random.randint(0, 10000, num_token_ids))

def test_insert(args) -> float:

    radix_cache = RadixCache(args.block_size)

    second_request = [generate_random_token_ids(args.block_size) for _ in range(args.num_blocks)]
    second_blocks = [PhysicalTokenBlock(None, -1, args.block_size, -1, -1) for i in range(args.num_blocks)]
    first_request = second_request[:math.ceil(args.num_blocks * args.cache_ratio)]
    first_blocks = second_blocks[:math.ceil(args.num_blocks * args.cache_ratio)]

    # Initialize the cache with the first request
    radix_cache.insert(first_request, first_blocks, None)

    start = time.perf_counter()
    radix_cache.insert(second_request, second_blocks, None)
    end = time.perf_counter()
    duration = (end - start) * 1000_000 # in us
    return duration

def test_match(args) -> float:

    radix_cache = RadixCache(args.block_size)

    second_request = [generate_random_token_ids(args.block_size) for _ in range(args.num_blocks)]
    second_blocks = [PhysicalTokenBlock(None, -1, args.block_size, -1, -1) for i in range(args.num_blocks)]
    first_request = second_request[:math.ceil(args.num_blocks * args.cache_ratio)]
    first_blocks = second_blocks[:math.ceil(args.num_blocks * args.cache_ratio)]

    # Initialize the cache with the first request
    radix_cache.insert(first_request, first_blocks, None)

    start = time.perf_counter()
    radix_cache.match_prefix(second_request)
    end = time.perf_counter()
    duration = (end - start) * 1000_000
    
    return duration

if __name__ == "__main__":

    args = get_args()
    insert_time = []
    match_time = []

    for _ in range(args.repeat_times):
        insert_time.append(test_insert(args))
        match_time.append(test_match(args))
    print('avg_insert_time,avg_match_time,std_insert_time,std_match_time')
    print(f'{np.mean(insert_time)},{np.mean(match_time)},{np.std(insert_time)},{np.std(match_time)}')


    