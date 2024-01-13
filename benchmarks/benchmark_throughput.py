"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Tuple
import math

from transformers import PreTrainedTokenizerBase

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[List[int], int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2
    ]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(prompt_token_ids)):
        tokenized_dataset.append((prompt_token_ids[1307], len(prompt_token_ids[1307]), len(completion_token_ids[1307])))
    
    # Filter out too long sequences.
    filtered_dataset: List[Tuple[List[int], int, int]] = []
    count = 0
    for prompt_token_ids, prompt_len, output_len in tokenized_dataset:
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt_token_ids, prompt_len, output_len))
        count += 1
        if count == num_requests:
            break

    # Sample the requests.
    # sampled_requests = filtered_dataset[:num_requests]
    #sampled_requests = random.sample(filtered_dataset, num_requests)
    return filtered_dataset #sampled_requests

def run_vllm(
    requests: List[Tuple[List[int], int, int]],
    model: str,
    tokenizer: str,
    tensor_parallel_size: int = 2,
    seed: int = 0,
    n: int = 1,
    use_beam_search: bool = False,
    max_num_seqs = 128,
    max_num_batched_tokens = 4096,
    split_two_phase = 1
) -> float:
    llm = LLM(
        model = model,
        tokenizer = tokenizer,
        tensor_parallel_size = tensor_parallel_size,
        seed = seed,
        max_num_seqs = max_num_seqs,
        max_num_batched_tokens = max_num_batched_tokens
    )
    start = time.time()

    for prompt_token_ids, prompt_len, output_len in requests:
        sampling_params = SamplingParams(
            n = n,
            temperature = 0.0 ,
            top_p = 1.0,
            use_beam_search = use_beam_search,
            ignore_eos = True,
            max_tokens = output_len,
        )
        # FIXME(woosuk): Do not use internal method.
        
        #resource_need = math.ceil((prompt_len + math.ceil(output_len / 200)) / 16)
        #resource_need = prompt_len + math.ceil(output_len / 200) * 200
        resource_need = math.ceil(output_len / 200) * 200
        #resource_need = math.ceil(resource_need / 16)
        predicted_len = prompt_len + math.ceil(output_len / 200) * 200
        '''if prompt_len >= 16:
            input_token_ids = prompt_token_ids[0: 16]
        else:
            input_token_ids = prompt_token_ids + [0] * (16 - prompt_len)'''
        llm._add_request(
            prompt = None,
            prompt_token_ids = prompt_token_ids, #input_token_ids,
            sampling_params=sampling_params,
            resoucre_need =  resource_need,
            predicted_len = predicted_len
        )


    # FIXME(woosuk): Do use internal method.
    llm._run_engine(use_tqdm = False, split_two_phase = split_two_phase)
    end = time.time()
        
    
    elapsed_time = end - start 
    return elapsed_time

def main(args: argparse.Namespace):
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = get_tokenizer(args.tokenizer)
    requests = sample_requests(args.dataset, args.num_prompts, tokenizer)

    
    elapsed_time = run_vllm(requests, args.model, args.tokenizer)
    
    total_num_tokens = sum(
        prompt_len + output_len
        for _, prompt_len, output_len in requests
    )

    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--dataset", type=str, default="/workspace/ShareGPT_V3_unfiltered_cleaned_split.json")
    parser.add_argument("--model", type=str, default="/workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--num-prompts", type=int, default=128,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model

    main(args)