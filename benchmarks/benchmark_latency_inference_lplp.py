"""Benchmark the latency of processing a single batch of requests."""
import argparse
import time
import random

import numpy as np
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams


def main(args: argparse.Namespace):
    print(args)

    # Process all the requests in a single batch if possible.
    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=4096,
    )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=0.0 if args.use_beam_search else 1.0,
        top_p=1.0,
        use_beam_search=args.use_beam_search,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)
    random.seed(42)
    prompt_token_ids = []
    
    dummy_prompt_token_ids = [[1] * args.input_len] * args.num_light
    prompt_token_ids.extend(dummy_prompt_token_ids)
    
    def run_to_completion(profile: bool = False):
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.time()
        #print(f"start at {start_time}")
        llm.generate(prompt_token_ids=prompt_token_ids,
                     sampling_params=sampling_params,
                     use_tqdm=False)

        end_time = time.time()
        latency = end_time - start_time
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        return latency

    print("Warming up...")
    run_to_completion(profile=False)

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(profile=False))
    print(f'Avg latency: {np.mean(latencies)} seconds')
    print(latencies)
    #print(f'Avg throughput: {args.input_len * 1  / np.mean(latencies):.2f} token/s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
                    'requests till completion.')
    parser.add_argument('--model', type=str, default='/workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=2)
    parser.add_argument('--input-len', type=int, default=512)
    parser.add_argument('--output-len', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--n', type=int, default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--num-light', type=int, default=1,
                        help='Number of generated sequences per heavy prompt.')
    
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters', type=int, default=1,
                        help='Number of iterations to run.')
    args = parser.parse_args()
    main(args)
