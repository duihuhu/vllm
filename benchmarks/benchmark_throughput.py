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
) -> List[Tuple[str, List[int], int]]:
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
    filtered_dataset: List[Tuple[str, List[int], int]] = []
    t = 0
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        #if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
        #    continue
        # if prompt_len > 1024 or prompt_len + output_len > 2048:
        if prompt_len >=2048 or output_len >=2048 or prompt_len + output_len >= 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_token_ids, output_len))
        t += 1
        if t == num_requests:
            break
    # Sample the requests.
    # sampled_requests = filtered_dataset[:num_requests]
    #sampled_requests = random.sample(filtered_dataset, num_requests)
    #return sampled_requests
    return filtered_dataset

def run_vllm(
    requests: List[Tuple[str, List[int], int]],
    model: str,
    tokenizer: str,
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    batch_size: int,
    split_two_phase: int
) -> float:
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        max_num_seqs=batch_size,
        max_num_batched_tokens = 4096
    )

    # Add the requests to the engine.
    for _, prompt_token_ids, output_len in requests:
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
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )

    #start = time.time()
    # FIXME(woosuk): Do use internal method.
    outputs = llm._run_engine(use_tqdm=False, split_two_phase=split_two_phase, filepath=args.file)
    #end = time.time()
    '''with open("/workspace/vllm/benchmarks/log_13b.txt", 'a') as file:
        for output in outputs:
            file.write(f"prompt {len(output.prompt_token_ids)}, output {len(output.outputs[0].token_ids)}\n")'''
    
    '''elapsed_time = end-start 
    total_num_tokens = sum(
        len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
        for output in outputs
    )
    print(f"End start is {start}, End end is {end}")
    print("total_num_reqs: ", len(outputs))
    print("total_num_tokens: ", total_num_tokens)
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
         f"{total_num_tokens / elapsed_time:.2f} tokens/s")'''

    #return end - start
    return -1


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
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = get_tokenizer(args.tokenizer)
    requests = sample_requests(args.dataset, args.num_prompts, tokenizer)

    if args.backend == "vllm":
        elapsed_time = run_vllm(
            requests, args.model, args.tokenizer, args.tensor_parallel_size,
            args.seed, args.n, args.use_beam_search, args.batch_size, args.split_two_phase)
    elif args.backend == "hf":
        assert args.tensor_parallel_size == 1
        elapsed_time = run_hf(requests, args.model, tokenizer, args.n,
                              args.use_beam_search, args.hf_max_batch_size)
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    '''total_num_tokens = sum(
        prompt_len + output_len
        for _, prompt_len, output_len in requests
    )
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend", type=str, choices=["vllm", "hf"],
                        default="vllm")
    parser.add_argument("--dataset", type=str, default="/workspace/ShareGPT_V3_unfiltered_cleaned_split.json")
    parser.add_argument("--model", type=str, default="/workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=2)
    parser.add_argument("--n", type=int, default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=128,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size", type=int, default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split-two-phase", type=int, default=1)
    parser.add_argument("--file", type=str, default="/workspace/vllm/benchmarks/bin200count128.txt")
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
