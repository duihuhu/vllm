from utils import get_args, set_seed
import asyncio
from transformers import AutoTokenizer
    
if __name__ == "__main__":

    args = get_args()
    set_seed(42)
    # tokenizer_path = "/data/zhaoyiyang/Llama-2-7B-fp16"
    tokenizer_path = "/home/jovyan/models/Llama-2-13b-hf"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if args.dataset == "ShareGPT":
        from dataset_processing.ShareGPT import sample_requests
        reqs, multi_conversations_range = sample_requests(
            # "/data/hujunhao/FlowServe/datasets/ShareGPT_V3_unfiltered_cleaned_split.json", 
            "/home/jovyan/hjh/datasets/ShareGPT_V3_unfiltered_cleaned_split.json", 
            tokenizer, 
            args.num_requests
        )
    elif args.dataset == "LooGLE":
        from dataset_processing.LooGLE import sample_requests
        reqs, multi_conversations_range = sample_requests(
            # ["/data/hujunhao/FlowServe/datasets/LooGLE/shortdep_qa.json", "/data/hujunhao/FlowServe/datasets/LooGLE/longdep_qa.json"], 
            # "/data/hujunhao/FlowServe/datasets/LooGLE/shortdep_qa.json",
            "/home/jovyan/hjh/datasets/LooGLE/shortdep_qa.json",
            tokenizer, 
            args.num_requests
        )
    elif args.dataset == "ReAct":
        from dataset_processing.ReAct import sample_requests
        reqs, multi_conversations_range = sample_requests(
            # "/data/hujunhao/FlowServe/datasets/hotpotqa_100.jsonl", 
            "/home/jovyan/hjh/datasets/hotpotqa_100.jsonl", 
            tokenizer, 
            args.num_requests
        )

    if args.test_type == "open":
        from test_type.open_loop import run
        asyncio.run(run(args, reqs))
    elif args.test_type == "closed":
        from test_type.closed_loop import run
        asyncio.run(run(args, reqs, multi_conversations_range))


    
