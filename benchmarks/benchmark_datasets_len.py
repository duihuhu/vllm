"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
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
    for i in range(len(dataset)):
        # if i < 1 :
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))
        # else:
        #     output_len = len(completion_token_ids[8])
        #     tokenized_dataset.append((prompts[8], prompt_token_ids[8], output_len))
    # print(prompts[71])
    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        # if prompt_len > 1024 or prompt_len + output_len > 2048:
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    # sampled_requests = filtered_dataset[:num_requests]
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests



def sample_requests_summary(dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    
    for data in dataset:
        prompt_token_ids = tokenizer(data["input"]).input_ids
        completion_token_ids = tokenizer(data["output"]).input_ids
        print("prompt len, prompt token len ,completions len ,completions token len: ", 
              len(data['input']), len(prompt_token_ids), len(data['output']), len(completion_token_ids))
    return 

def sample_requests_write(dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    
    for data in dataset:
        prompt_token_ids = tokenizer(data["input"]).input_ids
        completion_token_ids = tokenizer(data["response"]).input_ids
        print("prompt len, prompt token len ,completions len ,completions token len: ", 
              len(data['input']), len(prompt_token_ids), len(data['response']), len(completion_token_ids))
    return 

def main(args: argparse.Namespace):
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = get_tokenizer(args.tokenizer)
    # requests = sample_requests(args.dataset, args.num_prompts, tokenizer)
    sample_requests_summary(args.dataset, tokenizer)
    # sample_requests_write(args.dataset, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend", type=str, choices=["vllm", "hf"],
                        default="vllm")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n", type=int, default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size", type=int, default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--split-two-phase", type=int, default=0)
    args = parser.parse_args()

    if args.backend == "vllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF max batch size is required for HF backend.")
    if args.tokenizer is None:
        args.tokenizer = args.model

    main(args)
