import argparse

from vllm.chunked.chunkrunner import ChunkRunner
from vllm.transformers_utils.tokenizer import get_tokenizer

def main(args: argparse.Namespace):
    tokenizer = get_tokenizer(args.tokenizer)

    chunkrunner = ChunkRunner(tokenizer = tokenizer)

    model_name = args.model
    chunkrunner.set_self_model_config(model = model_name)

    chunk_size = args.chunk_size
    chunk_num = args.chunk_num
    chunkrunner.set_self_chunkworker(chunk_size = chunk_size, chunk_num = chunk_num)

    chunkrunner.set_inputs(dataset_path = args.dataset, num_requests = args.num_prompts)

    chunkrunner.run_worker()

    for seq_id, sequence in chunkrunner.chunk_worker.job_sequences.items():
        print(f"{seq_id}'s first output token is {sequence.first_token_str} while it's id is {sequence.first_token_id} \ 
              and it's prob is {sequence.first_token_logprob}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run vllm's prefill stage in chunk")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--num-prompts", type=int, default=10,
                        help="Number of prompts to process.")
    parser.add_argument("--chunk-szie", type=int, default=560)
    parser.add_argument("--chunk-num", type=int, default=100)
    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.model

    main(args)