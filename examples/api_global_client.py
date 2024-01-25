"""Example Python client for vllm.entrypoints.api_server"""
#start prefilled vllm: python3 -m vllm.entrypoints.api_server --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/  --host 127.0.0.1 --port 8000 --tensor-parallel-size 2
#start decode vllm:  python3 -m vllm.entrypoints.api_server --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/  --host 127.0.0.1 --port 8001 --tensor-parallel-size 2
#start workload: python3 prefill_decode_serving_client.py --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/  --num-prompts 4 --host 127.0.0.1 --port 9000

import argparse
import json
from typing import Iterable, List, Tuple, Optional

from transformers import PreTrainedTokenizerBase

from vllm.transformers_utils.tokenizer import get_tokenizer
import requests
import random
from vllm.utils import random_uuid

import fastapi
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
app = fastapi.FastAPI()
TIMEOUT_KEEP_ALIVE = 5  # seconds

import api_global_scheduer_config as cfg
import threading
session_event = {}

def post_inited_request(session_id: str, 
                        prompts: List[str],
                        request_id: str,
                        api_url: str,
                        n: int = 1) -> requests.Response:
    for prompt in prompts:
        request_id = random_uuid()
        headers = {"User-Agent": "Test Client"}
        pload = {
            # "user_id": user_id,
            "session_id": session_id,
            "prompt": prompt,
            "request_id": request_id,
            "n": n,
            "use_beam_search": False,
            "temperature": 0.0,
            # "max_tokens": 16,
            'ignore_eos': True,
        }
        response = requests.post(api_url, headers=headers, json=pload, stream=True)
        event = session_event[session_id]
        event.clear()
        event.wait()
    print("this thread mul consevration end ")
    return response

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, List[int], str]] :
    random.seed(0)
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
    # prompts = []
    # for prompt, _ in dataset:
    #     prompts.append("San Francisco is a")
    
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    # filtered_dataset: List[Tuple[str, int, int]] = []
    # filtered_prompts: List[str] = [] 
    # filtered_tokenids: List[str] = []
    filtered_dataset: List[Tuple[str, List[int], str, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        request_id = random_uuid()
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_token_ids, request_id, output_len))
        # filtered_prompts.append(prompt)
        # filtered_tokenids.append(prompt_token_ids)
    # Sample the requests.
    # sampled_requests = random.sample(filtered_dataset, num_requests)

    sampled_prompts = random.sample(filtered_dataset, num_requests)
    return sampled_prompts

@app.post("/response")
async def add_reqs(request: Request) -> Response:
    request_dict = await request.json()
    session_id = request.get("session_id")
    event = session_event[session_id]
    event.set()
    print(session_id, " recv response ")
    return 


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--session-num", type=int, default=1)

    args = parser.parse_args()
    
    if args.tokenizer is None:
        args.tokenizer = args.model
    tokenizer = get_tokenizer(args.tokenizer)
    session_ids = []
    for i in range(cfg.session_num):
        session_id = random_uuid()
        session_ids.append(session_id)
        event = threading.Event()
        session_event[session_id] = event
        
    prompts = ["What is the easiest idea to earn money", "What is the easiest idea to earn money"]
    
    task_td = []
    for session_id in session_ids:
        task_td.append(threading.Thread(target=post_inited_request, args=(session_id, prompts, cfg.add_reqs_url,  args.n)))
   
    for td in task_td:
      td.start()   
       
    uvicorn.run(app,
            host=cfg.host_ip,
            port=cfg.client_port,
            log_level="info",
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
    
    for td in task_td:
      td.join()  