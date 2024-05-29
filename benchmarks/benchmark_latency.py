"""Benchmark the latency of processing a single batch of requests."""
import argparse
import time
#import json
#import random

import numpy as np
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams
#from vllm.transformers_utils.tokenizer import get_tokenizer

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
        #max_num_batched_tokens=args.batch_size * args.input_len,
        max_num_batched_tokens = 4096
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

    '''sampling_params2 = SamplingParams(
        n=args.n,
        temperature=0.0 if args.use_beam_search else 1.0,
        top_p=1.0,
        use_beam_search=args.use_beam_search,
        ignore_eos=True,
        max_tokens=args.long_len,
    )'''

    '''tokenizer = get_tokenizer(args.model)
    with open("/workspace/ShareGPT_V3_unfiltered_cleaned_split.json") as f:
        dataset = json.load(f)
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2
    ]
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        if output_len > 2000:
            tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))
            break'''

    #andom.seed(0)
    inputs = []
    dummy_prompt_token_ids = [1] * args.input_len
    #dummy_prompt_token_ids2 = [1] * 48
    inputs.append(dummy_prompt_token_ids)
    inputs.append(dummy_prompt_token_ids)
    #for _ in range(10 - args.ratio):
    #    inputs.append(dummy_prompt_token_ids)
    #for _ in range(args.ratio):
    #    inputs.append(dummy_prompt_token_ids2)
    #inputs.append(dummy_prompt_token_ids)
    #inputs.append(dummy_prompt_token_ids2)
    #print(tokenized_dataset)
    #dummy_prompt_token_ids = []
    #dummy_prompt_token_ids.append(tokenized_dataset[0][1])
    def run_to_completion(profile: bool = False):
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.time()

        '''llm.generate(prompt_token_ids=inputs,
                     #prompt_token_ids=dummy_prompt_token_ids,
                     #prompt_token_ids=tokenized_dataset[0][1],
                     sampling_params=sampling_params,
                     use_tqdm=False,
                     filepath=args.filepath)'''
        '''for _ in range(2 - args.ratio):
            llm._add_request(prompt=None,
                             sampling_params=sampling_params,
                             prompt_token_ids=dummy_prompt_token_ids)
        for _ in range(args.ratio):
            llm._add_request(prompt=None,
                             sampling_params=sampling_params2,
                             prompt_token_ids=dummy_prompt_token_ids)'''
        llm._add_request(prompt=None,
                         sampling_params=sampling_params,
                         prompt_token_ids=dummy_prompt_token_ids)
        '''llm._add_request(prompt=None,
                         sampling_params=sampling_params,
                         prompt_token_ids=dummy_prompt_token_ids2)'''
        
        llm._run_engine(use_tqdm=False,split_two_phase=1, filepath=args.filepath, num=-1)

        end_time = time.time()
        latency = end_time - start_time
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        return latency

    print("Warming up...")
    run_to_completion(profile=False)
    print("End warming...")

    # Benchmark.
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        latencies.append(run_to_completion(profile=False))
    print(f'Avg latency: {np.mean(latencies)} seconds')


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
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters', type=int, default=5,
                        help='Number of iterations to run.')
    parser.add_argument('--filepath', type=str, default="/workspace/vllm/benchmarks/decode_ite4.txt")
    parser.add_argument('--ratio', type=int, default=0)
    parser.add_argument('--long-len', type=int, default=1024)
    args = parser.parse_args()
    main(args)