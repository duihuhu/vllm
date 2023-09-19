"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Tuple
import os

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.outputs import RequestOutput

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int
) -> List[Tuple[List, int, int]]:
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
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)

    # Padding and truncation
    prompts = []
    for prompt, _, _ in sampled_requests:
        prompts.append(prompt)
    prompts_ids = tokenizer(prompts, padding='max_length', truncation=True, max_length=max_length).input_ids
    result_requests = []
    for i, a_prompt_ids in enumerate(prompts_ids):
        result_requests.append((a_prompt_ids, len(a_prompt_ids), sampled_requests[i][2]))
    return result_requests


def run_vllm(
    requests: List[Tuple[List, int, int]],
    model: str,
    tokenizer: str,
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: int,
    max_num_seqs: int,
    input_len: int
):
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_seqs * input_len,
    )

    # Add the requests to the engine.
    for prompt_ids, _, output_len in requests:
        sampling_params = SamplingParams(
            n=n,
            temperature=0.0 if use_beam_search else 1.0,
            top_p=1.0,
            use_beam_search=use_beam_search,
            ignore_eos=True,
            max_tokens=output_len,
        )
        # FIXME(woosuk): Do not use internal method.
        llm._add_request(
            prompt=None,
            prompt_token_ids=prompt_ids,
            sampling_params=sampling_params,
        )

    start = time.time()
    # FIXME(woosuk): Do use internal method.
    the_outputs = llm._run_engine(use_tqdm=False)
    end = time.time()
    return end - start, the_outputs


def run_hf(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    n: int,
    use_beam_search: bool,
    max_batch_size: int,
) -> float:
    assert not use_beam_search
    llm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    if llm.config.model_type == "llama":
        # To enable padding in the HF backend.
        tokenizer.pad_token = tokenizer.eos_token
    llm = llm.cuda()

    pbar = tqdm(total=len(requests))
    start = time.time()
    batch: List[str] = []
    max_prompt_len = 0
    max_output_len = 0
    for i in range(len(requests)):
        prompt, prompt_len, output_len = requests[i]
        # Add the prompt to the batch.
        batch.append(prompt)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_output_len = max(max_output_len, output_len)
        if len(batch) < max_batch_size and i != len(requests) - 1:
            # Check if we can add more requests to the batch.
            _, next_prompt_len, next_output_len = requests[i + 1]
            if (max(max_prompt_len, next_prompt_len) + max(
                max_output_len, next_output_len)) <= 2048:
                # We can add more requests to the batch.
                continue

        # Generate the sequences.
        input_ids = tokenizer(batch, return_tensors="pt", padding=True).input_ids
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample=not use_beam_search,
            num_return_sequences=n,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_output_len,
        )
        # Include the decoding time.
        tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)
        pbar.update(len(batch))

        # Clear the batch.
        batch = []
        max_prompt_len = 0
        max_output_len = 0
    end = time.time()
    return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = get_tokenizer(args.tokenizer)
    requests = sample_requests(args.dataset, args.num_prompts, tokenizer, args.prompt_length)

    if args.backend == "vllm":
        back_tuple = run_vllm(
            requests, args.model, args.tokenizer, args.tensor_parallel_size,
            args.seed, args.n, args.use_beam_search, args.batch_size, args.prompt_length)
    elif args.backend == "hf":
        assert args.tensor_parallel_size == 1
        elapsed_time = run_hf(requests, args.model, tokenizer, args.n,
                              args.use_beam_search, args.hf_max_batch_size)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    elapsed_time = back_tuple[0]
    prefill_time = back_tuple[1][1] - back_tuple[1][0]
    decode_time = back_tuple[1][3] - back_tuple[1][2]
    requests_outputs = back_tuple[1][4]
    total_output_num_tokens = 0
    for outputs in requests_outputs:
        for output in outputs.outputs:
            total_output_num_tokens = total_output_num_tokens + len(output.token_ids)
    total_prompt_num_tokens = sum(prompt_len for _, prompt_len, _ in requests)
    total_num_tokens = total_prompt_num_tokens + total_output_num_tokens
    prefill_utilization = (2*1.25*100000000*args.batch_size*args.prompt_length) / (prefill_time*(args.batch_size/args.num_prompts)*14*1000000000000)
    #decode_utilization = (2*1.25*100000000*total_output_num_tokens) / (4*decode_time*14*1000000000000)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"Total: {total_num_tokens / elapsed_time:.2f} tokens/s, "
          f"Output: {total_output_num_tokens / elapsed_time:.2f} tokens/s, "
          f"Total prompts' number {len(requests)}, "
          f"Prompt tokens' number {total_prompt_num_tokens}, "
          f"Output tokens' number {total_output_num_tokens}, "
          f"Prefill GPU utilization {prefill_utilization:.2f}, "
          f"Decode GPU utilization - profiling..., "
          f"{back_tuple[1][0]},{back_tuple[1][1]},{back_tuple[1][2]},{back_tuple[1][3]}")
    
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
    parser.add_argument("--use-beam-search", type=int, default=0,
                        help="Whether to use beam serach")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size", type=int, default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument("--prompt-length", type=int, default=None, 
                        help="The token length of every prompt within a batch")
    parser.add_argument("--batch-size", type=int, default=None, 
                        help="Max seq number within a batch")
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
