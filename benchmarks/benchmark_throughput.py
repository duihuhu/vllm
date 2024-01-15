"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Tuple
import math
import numpy as np

from transformers import PreTrainedTokenizerBase

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer

def error() -> int:
    acc_value = ["0", "1"]
    acc_probs = [0.719, 0.281]

    gap_value = ["1lt", "1gt", "2lt", "2gt", "3lt", "3gt", "4lt", "4gt", "5lt", "5gt", "6lt", "7lt", "8lt", "9lt"]
    gap_probs = [0.73295 * (1-0.43798), 0.73295 * 0.43798, 0.16080 * (1-0.35689), 0.16080 * 0.35689, 0.06136 * (1-0.23148), 
                 0.06136 * 0.23148, 0.02500 * (1-0.18182), 0.02500 * 0.18182, 0.01136 * (1-0.05000), 0.01136 * 0.05000,
                 0.00625, 0.00114, 0.00057, 0.00057]
    
    random_acc = np.random.choice(acc_value, p = acc_probs)
    if random_acc == "1":
        random_gap = np.random.choice(gap_value, p = gap_probs)
        if random_gap == "1lt":
            return -1
        elif random_gap == "1gt":
            return 1   
        elif random_gap == "2lt":
            return -2
        elif random_gap == "2gt":
            return 2
        elif random_gap == "3lt":
            return -3     
        elif random_gap == "3gt":
            return 3
        elif random_gap == "4lt":
            return -4
        elif random_gap == "4gt":
            return 4
        elif random_gap == "5lt":
            return -5
        elif random_gap == "5gt":
            return 5
        elif random_gap == "6lt":
            return -6
        elif random_gap == "7lt":
            return -7
        elif random_gap == "8lt":
            return -8
        elif random_gap == "9lt":
            return -9
        else:
            return 10
    else:
        return 10

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
        tokenized_dataset.append((prompt_token_ids[i], len(prompt_token_ids[i]), len(completion_token_ids[i])))
    
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
    max_num_seqs = 256,
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
        e = error()
        resource_need = math.floor(output_len / 200) * 200
        if e != 10:
            resource_need += e * 200
            if resource_need > 2000:
                resource_need = 2000
            if resource_need < 200:
                resource_need = 200
        resource_need = math.ceil(resource_need / 16)
        # predicted_len = prompt_len + math.ceil(output_len / 200) * 200
        predicted_len = prompt_len + math.ceil(output_len / 200) * 200

        if prompt_len >= 80:
            input_token_ids = prompt_token_ids[0: 80]
        else:
            input_token_ids = prompt_token_ids + [0] * (80 - prompt_len)
        llm._add_request(
            prompt = None,
            prompt_token_ids = input_token_ids,
            sampling_params=sampling_params,
            resoucre_need =  resource_need,
            predicted_len = predicted_len
        )


    # FIXME(woosuk): Do use internal method.
    outputs = llm._run_engine(use_tqdm = False, split_two_phase = split_two_phase)
    end = time.time()

    t = 0
    for i, output in enumerate(outputs):
        end_length = len(output.outputs[0].token_ids)
        if end_length < requests[i][2]:
            t += 1
    
    elapsed_time = end - start
    print(f"{t} seqs have returned before output length")
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
    parser.add_argument("--num-prompts", type=int, default=256,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model

    main(args)