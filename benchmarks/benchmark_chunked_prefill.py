# benchmark for chunked prefill
import argparse
import json
import random
import time
from typing import List, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.outputs import RequestOutput

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
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[List[int], int, int]] = []
    num_requests_count = 0
    for _, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        '''if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue'''
        if prompt_len < 560:
            continue
        filtered_dataset.append((prompt_token_ids, prompt_len, output_len))
        num_requests_count += 1
        if num_requests_count == num_requests:
            break

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests

def run_chunked_vllm(
    requests: List[Tuple[List[int], int, int]],
    model: str,
    tokenizer: str,
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    batch_size: int,
    chunk: int
):
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        max_num_seqs=batch_size,
    )

    # Add the requests to the engine.
    offset = 0
    last_slot_num = 0
    for prompt_token_ids, prompt_len, output_len in requests:
        chunked_token_ids = []
        
        while True:
            st = offset * chunk
            ed = st + chunk
            if ed <= prompt_len:
                chunked_token_ids.append(prompt_token_ids[st: ed])
            else:
                last_slot_num = prompt_len - st
                chunked_token_ids.append(prompt_token_ids[st: prompt_len])
                break
            offset += 1
        
        sampling_params = SamplingParams(
            n=n,
            temperature=0.0 if use_beam_search else 1.0,
            top_p=1.0,
            use_beam_search=use_beam_search,
            ignore_eos=True,
            max_tokens=output_len,
        )
        # add chunks
        for chunk_prompt_token_ids in chunked_token_ids:
            llm._add_request(
                prompt=None,
                prompt_token_ids=chunk_prompt_token_ids,
                sampling_params=sampling_params
            )
        # add total prompt for verify
        llm._add_request(
            prompt=None,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params
        )

    start_time = time.time()
    _, hidden_states, total_hidden_states = llm._run_engine_in_chunk(use_tqdm=False, chunked_num = offset, chunked_size = chunk,
                                                      last_slot_num = last_slot_num)
    end_time = time.time()
    time_slot = end_time - start_time

    print(f"The shape of result tensor is {hidden_states.shape}")
    print(f"The shape of total result tensor is {total_hidden_states.shape}")
    print(total_hidden_states.eq(hidden_states))
    print(f"The prefill throughtput of chunked prompt is {len(requests[0][0]) / time_slot:.2f}")
    
    return 

def main(args: argparse.Namespace):
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = get_tokenizer(args.tokenizer)
    requests = sample_requests(args.dataset, args.num_prompts, tokenizer)
    
    run_chunked_vllm(
            requests, args.model, args.tokenizer, args.tensor_parallel_size,
            args.seed, args.n, args.use_beam_search, args.batch_size, args.chunk)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the chunked prefill throughput.")
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
    parser.add_argument("--chunk", type=int, default=560)
    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.model

    main(args)