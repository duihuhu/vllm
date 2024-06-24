from utils import set_seed
import asyncio
from transformers import AutoTokenizer
import argparse
import sys


def get_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--request-rate", type=float, default=20)
    parser.add_argument("--num-requests", type=int, default=512)
    parser.add_argument("--input-len", type=int, default=1)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="ShareGPT", choices=["ShareGPT", "LooGLE", "ReAct"])
    parser.add_argument("--test-type", type=str, default="open", choices=["open", "closed"])
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--duration", type=int, default=10)

    args = parser.parse_args()

    return args
    
if __name__ == "__main__":

    args = get_args()
    set_seed(42)
    tokenizer_path = "/data/zhaoyiyang/Llama-2-7B-fp16"
    # tokenizer_path = "/home/jovyan/models/Llama-2-13b-hf"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if args.dataset == "ShareGPT":
        from dataset_processing.ShareGPT import sample_requests
        reqs, multi_conversations_range = sample_requests(
            "/data/hujunhao/FlowServe/datasets/ShareGPT_V3_unfiltered_cleaned_split.json", 
            # "/home/jovyan/hjh/datasets/ShareGPT_V3_unfiltered_cleaned_split.json", 
            tokenizer, 
            args.num_requests
        )
    elif args.dataset == "LooGLE":
        from dataset_processing.LooGLE import sample_requests
        reqs, multi_conversations_range = sample_requests(
            # ["/data/hujunhao/FlowServe/datasets/LooGLE/shortdep_qa.json", "/data/hujunhao/FlowServe/datasets/LooGLE/longdep_qa.json"], 
            "/data/hujunhao/FlowServe/datasets/LooGLE/shortdep_qa.json",
            # "/home/jovyan/hjh/datasets/LooGLE/shortdep_qa.json",
            tokenizer, 
            args.num_requests
        )
    elif args.dataset == "ReAct":
        from dataset_processing.ReAct import sample_requests
        reqs, multi_conversations_range = sample_requests(
            "/data/hujunhao/FlowServe/datasets/hotpotqa_100.jsonl", 
            # "/home/jovyan/hjh/datasets/hotpotqa_100.jsonl", 
            tokenizer, 
            args.num_requests
        )

    if args.test_type == "open":
        from test_type.open_loop import run
        asyncio.run(run(args, reqs, multi_conversations_range))
    elif args.test_type == "closed":
        print(f'maximum number of clients: {len(multi_conversations_range)}', file=sys.stderr)
        assert args.num_clients <= len(multi_conversations_range), 'Number of clients should be less than or equal to the number of sessions'
        from test_type.closed_loop import run
        asyncio.run(run(args, reqs, multi_conversations_range))


    
