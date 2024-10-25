"""Example Python client for vllm.entrypoints.api_server"""
# python3 mul_vllm_serving_client.py --dataset /workspace/ShareGPT_V3_unfiltered_cleaned_split.json --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/  --num-prompts 200 --num-servers 2
#python3 -m vllm.entrypoints.api_server --model /workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/  --host 127.0.0.1 --port 8001 --tensor-parallel-size 2

import argparse
import json
from typing import Iterable, List, Tuple, Optional

from transformers import PreTrainedTokenizerBase

import fastapi
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.transformers_utils.tokenizer import get_tokenizer
import requests
import random
import threading
from vllm.utils import random_uuid
import time
import asyncio
import uvicorn
import multiprocessing
manager = multiprocessing.Manager()

prefilled_event = manager.Event()

app = fastapi.FastAPI()
TIMEOUT_KEEP_ALIVE = 5  # seconds
request_prompts_token_ids = {}
request_prompts = {}

mdecode_prefilled_num = 0
total_requests = 16
# batch_size = 4

# prefilled_event = asyncio.Event()
def clear_line(n: int = 1) -> None:
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    for _ in range(n):
        print(LINE_UP, end=LINE_CLEAR, flush=True)

def post_init_decode_prefill(prompts: List[str],
                      output_lens: List[int],
                      request_ids: List[str],
                      api_url: str,
                      n: int = 1,
                      stream: bool = False) -> requests.Response: 
    headers = {"User-Agent": "Test Client"}
    pload = {
        "prompts": prompts,
        "output_lens": output_lens,
        "request_ids": request_ids,
        "n": 1,
        "use_beam_search": False,
        "temperature": 0.0,
        # "max_tokens": 16,
        'ignore_eos': True,
        "stream": stream,
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response

def post_prefill_execute(prompts: List[str],
                      output_lens: List[int],
                      request_ids: List[str],
                      api_url_execute_prefill: str,
                      api_url_add_prefill: str,
                      n: int = 1,
                      stream: bool = False):
    print("start post_prefill_execute ")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # 运行异步函数
    loop.run_until_complete(post_prefill_exec(prompts, output_lens, request_ids, api_url_execute_prefill, api_url_add_prefill, n, stream))
    return

async def post_prefill_exec(prompts: List[str],
                      output_lens: List[int],
                      request_ids: List[str],
                      api_url_execute_prefill: str,
                      api_url_add_prefill: str,
                      n: int = 1,
                      stream: bool = False):
    global prefilled_event
    prefilled_event.wait()
    print("after prefilled_event wait ")
    num_prompts = len(prompts)
    batch_size = 16
    alread_send = 0
    while alread_send <= num_prompts:
        # if alread_send == 0:
        #     mprefill_status = "mprefill_execute"
        #     api_url = api_url_execute_prefill
        # else:
        mprefill_status = "mprefill_add "
        host = "127.0.0.1"
        api_url = f"http://{host}:{9000}/global_prefill_req_pool"

        headers = {"User-Agent": "Test Client"}
        pload = {
            "prompts": prompts[alread_send:alread_send + batch_size],
            "output_lens": output_lens[alread_send:alread_send + batch_size],
            "request_ids": request_ids[alread_send:alread_send + batch_size],
            "n": 1,
            "use_beam_search": False,
            "temperature": 0.0,
            # "max_tokens": 16,
            'ignore_eos': True,
            "stream": stream,
            "mprefill_status": mprefill_status
        }
        print("mprefill send ", api_url, " alread_send " , alread_send, time.time())
        response = requests.post(api_url, headers=headers, json=pload, stream=True)
        alread_send = alread_send + batch_size
        if alread_send >= num_prompts:
            break
        elif alread_send < num_prompts:
            if (alread_send + batch_size) > num_prompts:
                batch_size = num_prompts - alread_send
        time.sleep(2)
    return

def receive_mdecode_prefilled_signal(host, port):
    uvicorn.run(app,
            host=host,
            port=port,
            log_level="info",
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
    
#background threads
@app.post("/mdecode_prefilled")
async def mdecode_prefilled(request: Request) -> Response:
    global total_requests
    global mdecode_prefilled_num
    global prefilled_event
    request_dict = await request.json()
    prefilled_num = request_dict.pop("prefilled_num")
    mdecode_prefilled_num = mdecode_prefilled_num + prefilled_num
    if mdecode_prefilled_num == total_requests:
        print("controller already recv prefilled signal ", mdecode_prefilled)
        prefilled_event.set()
    return

def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    for chunk in response.iter_lines(chunk_size=8192,
                                     decode_unicode=False,
                                     delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            output = data["text"]
            yield output


def get_response(response: requests.Response) -> List[str]:
    data = json.loads(response.content)
    output = data["text"]
    return output

def sample_requests_aplaca(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    aplaca_data = []
    for data in dataset:
        if data["input"] == "":
            aplaca_data.append((data["instruction"], data["output"]))

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in aplaca_data]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in aplaca_data]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for p, pt, c, ct in zip(prompts, prompt_token_ids, completions, completion_token_ids):
        tokenized_dataset.append(len(p), len(pt), len(ct))

    # Filter out too long sequences.
    # filtered_dataset: List[Tuple[str, int, int]] = []
    filtered_dataset: List[Tuple[str, List[int], str, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        request_id = random_uuid()
        prompt_len = len(prompt_token_ids)
        # if prompt_len > 256 or output_len > 128:
        if prompt_len < 4 or output_len < 4:
        # if prompt_len < 512 or output_len < 256:
        # if prompt_len > 512 or output_len < 128:
            # Prune too short sequences.
            continue
        # if prompt_len > 1024 or prompt_len + output_len > 2048:
        if prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_token_ids, request_id, output_len))
        # filtered_prompts.append(prompt)
        # filtered_tokenids.append(prompt_token_ids)
    # Sample the requests.
    # sampled_requests = random.sample(filtered_dataset, num_requests)

    sampled_prompts = random.sample(filtered_dataset, num_requests)
    return sampled_prompts

     
def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[str]:
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
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        # output_len = 16
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    # filtered_dataset: List[Tuple[str, int, int]] = []
    filtered_dataset: List[Tuple[str, List[int], str, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        request_id = random_uuid()
        prompt_len = len(prompt_token_ids)
        # if prompt_len > 256 or output_len > 128:
        if prompt_len < 4 or output_len < 4:
        # if prompt_len < 512 or output_len < 256:
        # if prompt_len > 512 or output_len < 128:
            # Prune too short sequences.
            continue
        # if prompt_len > 1024 or prompt_len + output_len > 2048:
        if prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_token_ids, request_id, output_len))
        # filtered_prompts.append(prompt)
        # filtered_tokenids.append(prompt_token_ids)
    # Sample the requests.
    # sampled_requests = random.sample(filtered_dataset, num_requests)

    sampled_prompts = random.sample(filtered_dataset, num_requests)
    return sampled_prompts



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--prompt", type=str, default="San Francisco is a")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--num-servers", type=int, default=1)

      
    args = parser.parse_args()
    
    if args.tokenizer is None:
        args.tokenizer = args.model
    tokenizer = get_tokenizer(args.tokenizer)
    
    sampled_prompts = sample_requests_aplaca(args.dataset, args.num_prompts, tokenizer)
    prompts = []
    request_ids = []
    output_lens = []
    for prompt in sampled_prompts:
      request_prompts[prompt[-2]] = prompt[0]
      request_prompts_token_ids[prompt[-2]] = prompt[-3]
      prompts.append(prompt[0])
      request_ids.append(prompt[-2])
      output_lens.append(prompt[-1])
      print(len(prompt[1]), prompt[-1])
    
    n = args.n
    stream = args.stream
            
    # while True:
    #     if status == 0:
    #         host_decode = args.host
    #         port_decode = args.port + 1
    #         api_url = f"http://{host_decode}:{port_decode}/init_decode_prefill"
    #         response = post_init_decode_prefill(prompts, output_lens, request_ids, api_url , n, stream)
    #         status = status + 1
    #     elif status == 2:
    #         host_prefill = args.host
    #         port_prefill = args.port
    #         api_url = f"http://{host_prefill}:{port_prefill}/prefilled"
    #         post_prefill_execute(prompts, output_lens, api_url, n, stream)

    #         break

    host_decode = args.host
    port_decode = args.port - 1000 + 2 
    api_url_decode = f"http://{host_decode}:{port_decode}/init_mdecode"
    task_td = []
    task_td.append(threading.Thread(target=post_init_decode_prefill, args=(prompts, output_lens, request_ids, api_url_decode , n, stream)))
    
    host_prefill = args.host
    port_prefill = args.port - 1000 + 1
    api_url_execute_prefill = f"http://{host_prefill}:{port_prefill}/mprefill_execute"
    api_url_add_prefill = f"http://{host_prefill}:{port_prefill}/mprefill_add"

    task_td.append(threading.Thread(target=post_prefill_execute, args=(prompts, output_lens, request_ids, api_url_execute_prefill, api_url_add_prefill, n, stream)))
      
    task_td.append(threading.Thread(target=receive_mdecode_prefilled_signal, args=(args.host, args.port)))
    
    for td in task_td:
      td.start()
    for td in task_td:
      td.join()  
      

    # if stream:
    #     num_printed_lines = 0
    #     for h in get_streaming_response(response):
    #         clear_line(num_printed_lines)
    #         num_printed_lines = 0
    #         for i, line in enumerate(h):
    #             num_printed_lines += 1
    #             print(f"Beam candidate {i}: {line!r}", flush=True)
    # else:
    #     output = get_response(response)
    #     for i, line in enumerate(output):
    #         print(f"Beam candidate {i}: {line!r}", flush=True)

