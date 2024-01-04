import argparse
import json
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
import asyncio
import threading
import time
import mmap
import os
import requests
import random
from vllm.transformers_utils.tokenizer import get_tokenizer
from typing import List, Tuple
# import multiprocessing
# manager = multiprocessing.Manager()

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()

mdecode_status = "init_mdecode_prefill"
 
decode_event = threading.Event()

dp_md = 'mdispatcher_to_mdecode.txt'

# if not os.path.isfile(dp_md):
    # create initial file
with open(dp_md, "w+b") as fd:
    fd.write(b'\x00' * 35 * 1024)

# init   
fd = open(dp_md, "r+b")
mm = mmap.mmap(fd.fileno(), 35 * 1024, access=mmap.ACCESS_WRITE, offset=0)

# @app.post("/notify_mdecode")
def notify_mdecode():
    global decode_event
    global mdecode_status
    hex_char = b'\x0F'
    # 判断内存
    prefill_nums = b'\x00'
    request_num =  b'\x00'
    already_num = 0 
    # 读取内存映射区域的数据

    while True:
        if prefill_nums != mm[(already_num*35+34):(already_num*35+35)]:
            # prefill_nums = mm[(already_num*35):(already_num*35+1)]
            # start = time.time()
            request_num = int.from_bytes(mm[(already_num*35):(already_num*35+1)], byteorder='big')
            request_id = mm[(already_num*35+1):(already_num*35+33)].decode("utf-8")
            label = int.from_bytes(mm[(already_num*35+33):(already_num*35+34)], byteorder='big')
            arrive_time = time.time()
            # print("decode get data " , request_id, start, end, end-start)
            # print("mdecode recv signal from mprefill ", time.time())
            # print("request info ", request_id, request_num, label, time.time())
            # if request_num > 0:
            # engine.convert_reqs_status_by_num(request_num)
            # engine.convert_reqs_status(request_id)
            engine.convert_req_label_status(request_id, label)
            add_time = time.time()
            print("decode get data " , request_id, arrive_time, add_time, add_time-arrive_time)

            mdecode_status = "decode"
            already_num = already_num + 1
            decode_event.set()

def init_mdecode_prefill():
    global mdecode_status
    while True:
        # print("init_mdecode_prefill ", mdecode_status)
        if mdecode_status == "init_mdecode_prefill":
            decode_event.wait()
            results_generator = engine.generate_mdecode_prefill()
        elif mdecode_status == "decode":
            print("status is chanage, mdecode start exec decode", mdecode_status)
            engine.generate_decode()
        decode_event.clear()
        decode_event.wait()
        

def post_monitor_request(monitor_url: str,
                      host: str,
                      service_port: int,
                      machine_type: str, 
                      num_labels: int ,
                      ) -> requests.Response:
    headers = {"User-Agent": "mdecode "}
    timestamp = time.time()
    pload = {
        "host": host,
        "service_port": service_port,
        "machine_type": machine_type,
        "num_labels": num_labels,
        "timestamp":timestamp,
    }
    response = requests.post(monitor_url, headers=headers, json=pload)
    
    return response

def post_mdecode_info(host, service_port, machine_type, num_labels):
    monitor_url = "http://127.0.0.1:9000/mdecode_monitor_report"
    response = post_monitor_request(monitor_url, host, service_port, machine_type, num_labels)

    
def monitor_mdecode_info(host, service_port):
    global engine
    machine_type = "decode"
    while True:
        num_labels = engine.monitor_mdecode_info()
        post_mdecode_info(host, service_port, machine_type, num_labels)
        time.sleep(1000)


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
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
    for req in sampled_requests:
        print("choose req info ", req[1], req[2])
    return sampled_requests

def post_request_id(request_ids
                      ) -> requests.Response:
    request_url = "http://127.0.0.1:8000/mdecode_requestIds"
    headers = {"User-Agent": "request_id "}
    pload = {
        "request_ids": request_ids,
    }
    response = requests.post(request_url, headers=headers, json=pload)
    
def init_mdecode(engine, num_prompts):
    """init mdecode machine before execute. 
    add request to queue
    """
    tokenizer = get_tokenizer("/workspace/opt-125m")
    sampled_requests = sample_requests("/workspace/ShareGPT_V3_unfiltered_cleaned_split.json", num_prompts, tokenizer)
    request_ids = []
    for prompt, _, output_len in sampled_requests:
        request_id = random_uuid()
        sampling_params = SamplingParams(
            n=1,
            # temperature=0.0 if use_beam_search else 1.0,
            temperature=0.0,
            top_p=1.0,
            use_beam_search=False,
            ignore_eos=True,
            max_tokens=output_len,
        )
        engine.engine.add_request(request_id, prompt, sampling_params)
        request_ids.append(request_id)
    post_request_id(request_ids)
    decode_event.set()
    print("init_mdecode return ")
    ret = {"text": 'test'}
    return JSONResponse(ret)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-prompts", type=int, default=128)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    td_list= []
    t1 = threading.Thread(target=init_mdecode_prefill, args=(engine,))
    td_list.append(t1)
    t2 = threading.Thread(target=notify_mdecode)
    td_list.append(t2)
    t3 = threading.Thread(target=monitor_mdecode_info, args=(args.host, args.port))
    td_list.append(t3)

    for td in td_list:
        td.start()

    init_mdecode(engine, args.num_prompts)
    
    for td in td_list:
        td.join()
    