import argparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import uvicorn

from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.chunked.chunkrunner import ChunkRunner, RequestInfo
from vllm.chunked.chunk import ChunkSamplingParams

from typing import List
import threading
import time
import mmap
import os
import requests
# import multiprocessing
# manager = multiprocessing.Manager()

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.

#to record request && label
request_label = {}

request_event = {}

mdecode_info = {}
prefill_sched_batch = 16
app = FastAPI()

mp_dp = 'mprefill_to_mdispatcher.txt'

# if not os.path.isfile(mp_dp):
    # create initial file
with open(mp_dp, "w+b") as fd:
    fd.write(b'\x00' * 35 * 1024)

mp_md_dp = 'mprefill_mdispatcher_to_mdecode_mdispatcher.txt'

# if not os.path.isfile(mp_md_dp):
#     # create initial file
with open(mp_md_dp, "w+b") as fd:
    fd.write(b'\x00' * 35 * 1024)

prefill_event = threading.Event()

predict_event = threading.Event()

fd = open(mp_dp, "r+b")
mm = mmap.mmap(fd.fileno(), 35 * 1024, access=mmap.ACCESS_WRITE, offset=0)

def mmap_warm():
    p_num = 1
    num = 0
    request_id = "00000000000000000000000000000000"
    label = 0 
    combined_info_bytes = p_num.to_bytes(1, byteorder='big') + num.to_bytes(1, byteorder='big') + request_id.encode("utf-8") + label.to_bytes(1, byteorder='big')
    print("combined_info_bytes ", len(combined_info_bytes))
    mm.seek(0)
    mm.write(combined_info_bytes)
    time.sleep(1)
    mm.seek(0)
    mm.write(b'\x00' * 35)
    return

def mprefill_exec_prefill(request_label, request_event):
    # with open(mp_dp, "r+b") as fd:
    #     mm = mmap.mmap(fd.fileno(), 35, access=mmap.ACCESS_WRITE, offset=0)
    prefill_nums = 0 
    while True:
        prefill_event.wait() 
        execute_time = time.time()
        print("prefill start execute time 1", execute_time)
        #prefill_nums = engine.mprefill_generate_prefill(mm, prefill_nums)
        prefill_nums = chunkrunner.mprefill_generate_prefill(mm, prefill_nums, request_label, mdecode_info, request_event, prefill_sched_batch)
        prefill_event.clear()
        prefill_event.wait()     
        # mm.close()
        # fd.close()
        
async def mprefill_add_prefill(request_dict):
    print("mprefill add prefill request ", time.time())
    request_ids = request_dict.pop("request_ids")
    prompts = request_dict.pop("prompts")
    output_lens = request_dict.pop("output_lens")
    stream = request_dict.pop("stream", False)
    mprefill_status = request_dict.pop("mprefill_status")
    n = request_dict.pop("n")
    use_beam_search = request_dict.pop("use_beam_search")
    temperature = request_dict.pop("temperature")
    ignore_eos = request_dict.pop("ignore_eos")
    prompts_token_ids_s: List[List[int]] = []
    sampling_params_s: List[sampling_params] = []
    
    for request_id, prompt in zip(request_ids, prompts):
        prompt_token_ids = tokenizer(prompt).input_ids
        # prompts_token_ids_s.append(prompt_token_ids) 
        sampling_params = ChunkSamplingParams(temperature = temperature, top_p = 1.0, top_k = -1)
        chunkrunner.request_waiting[0].append(request_id)
        chunkrunner.request_waiting[1].append(prompt_token_ids)
        chunkrunner.request_waiting[2].append(sampling_params)
        chunkrunner.request_waiting[3].append(prompt)
        chunkrunner.request_waiting[4].append(len(prompt_token_ids))
        # request_info = RequestInfo(input_len=len(prompt_token_ids))
        # chunkrunner.request_info_waiting.append(request_info)
        event = threading.Event()
        event.clear()
        request_event[request_id] = event

    # for _ in range(len(prompts_token_ids_s)):
    #     sampling_params = ChunkSamplingParams(temperature = temperature, top_p = 1.0, top_k = -1)
    #     sampling_params_s.append(sampling_params)
    
    if len(chunkrunner.request_waiting[0]) >=  prefill_sched_batch:
    #sampling_params_list = []
    #for i in range(len(prompts)):
    #    sampling_params = SamplingParams(**request_dict)
    #    sampling_params_list.append(sampling_params)
    # engine.add_request(prompts=prompts, output_lens=output_lens, request_ids=request_ids, sampling_params=sampling_params_list)
    #engine.add_mprefill_request(prompts=prompts, output_lens=output_lens, request_ids=request_ids, sampling_params=sampling_params_list)
        execute_time = time.time()
        print("prefill start execute time ", execute_time)
        for i in range(4):
            # start_sort_time  = time.time()
            sorted_request_waiting = sorted(zip(chunkrunner.request_waiting[0][i*16:(i+1)*16], chunkrunner.request_waiting[1][i*16:(i+1)*16],
                                    chunkrunner.request_waiting[2][i*16:(i+1)*16], chunkrunner.request_waiting[3][i*16:(i+1)*16],
                                    chunkrunner.request_waiting[4][i*16:(i+1)*16]), key=lambda x: x[4])
            sort0 , sort1, sort2, sort3, sort4 = zip(*sorted_request_waiting)
            chunkrunner.request_waiting[0][i*16:(i+1)*16] = sort0
            chunkrunner.request_waiting[1][i*16:(i+1)*16] = sort1
            chunkrunner.request_waiting[2][i*16:(i+1)*16] = sort2
            chunkrunner.request_waiting[3][i*16:(i+1)*16] = sort3
            chunkrunner.request_waiting[4][i*16:(i+1)*16] = sort4
            # end_sort_time  = time.time()
            # print("sort execute time ", end_sort_time-start_sort_time)
            
            # start_sort_time  = time.time()
            # chunkrunner.request_info_waiting.sort(key=lambda x: x.input_len)
            # end_sort_time  = time.time()
            # print("class sort execute time ", end_sort_time - start_sort_time)
            
        # chunkrunner.request_waiting[0],  chunkrunner.request_waiting[1] ,  chunkrunner.request_waiting[2], chunkrunner.request_waiting[3] , chunkrunner.request_waiting[4] =zip(*sorted_request_waiting)
        execute_time_end = time.time()
        # print("sort time  ", execute_time_end - execute_time)
        # print("chunkrunner ", chunkrunner.request_waiting[4])

        chunkrunner.add_requests_to_job_sequences(prompts_s = chunkrunner.request_waiting[3], prompt_token_ids_s = chunkrunner.request_waiting[1], 
                                                sampling_params_s = chunkrunner.request_waiting[2], request_ids=chunkrunner.request_waiting[0], request_label=request_label)
        
        chunkrunner_125m.add_requests_125m_sequences(chunkrunner.request_waiting[0], chunkrunner.request_waiting[3], chunkrunner.request_waiting[4])
        
        chunkrunner.request_waiting[0] = []
        chunkrunner.request_waiting[1] = []
        chunkrunner.request_waiting[2] = []
        chunkrunner.request_waiting[3] = []
        chunkrunner.request_waiting[4] = []
        
        predict_event.set()
        prefill_event.set()

@app.post("/mprefill_add")
async def mprefill_add(request: Request) -> Response:
    request_dict = await request.json()
    await mprefill_add_prefill(request_dict)
    ret = {"text": 'test'}
    return JSONResponse(ret)

@app.on_event("startup")
def startup_decode_event():
    threading.Thread(target=mprefill_exec_prefill, args=(request_label, request_event), daemon=True).start()
    threading.Thread(target=monitor_mprefill_info, args=(args.host, args.port) ,daemon=True).start()

def post_monitor_request(monitor_url: str,
                      host: str,
                      service_port: int,
                      machine_type: str, 
                      unfinished_req: int ,
                      unfinished_tokens: int
                      ) -> requests.Response:
    headers = {"User-Agent": "mprefill "}
    timestamp = time.time()
    pload = {
        "host": host,
        "service_port": service_port,
        "machine_type": machine_type,
        "unfinished_req": unfinished_req,
        "unfinished_tokens": unfinished_tokens,
        "timestamp":timestamp,
    }
    response = requests.post(monitor_url, headers=headers, json=pload)
    
    return response

def post_mprefill_info(host, service_port, machine_type, unfinished_req, unfinished_tokens):
    monitor_url = "http://127.0.0.1:9000/mprefill_monitor_report"
    response = post_monitor_request(monitor_url, host, service_port, machine_type, unfinished_req, unfinished_tokens)
    data = response.json()
    for key, value in data.items():
        mdecode_info[key] = value
        
    # print("mdecode_info: ", mdecode_info)
    
def monitor_mprefill_info(host, service_port):
    global chunkrunner
    machine_type = "prefill"
    while True:
        unfinished_tokens = chunkrunner.monitor_mprefill_info()
        # print("unfinished_tokens ", unfinished_tokens)
        post_mprefill_info(host, service_port, machine_type, 0, unfinished_tokens)
        time.sleep(1000)
    return

def execute_13b_model():
    global chunkrunner
    print("warm up...")
    chunkrunner.run_worker()
    print("end warm up")
    return 

def execute_125m_model():
    global chunkrunner_125m
    global request_label
    global predict_event
    global request_event
    chunkrunner_125m.warmup(request_label)
    while True:
        print("predict model start ")
        predict_event.wait() 
        execute_time = time.time()
        print("predict start execute time ", execute_time)
        # chunkrunner_125m.execute_predict(request_label, request_event, prefill_sched_batch)
        chunkrunner_125m.execute_predict(request_label, request_event)
        predict_event.clear()
        predict_event.wait()     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--model", type=str, default="/workspace/models/facebook/opt-125m")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-num", type=int, default=50)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--predict-model", type=str, default="/workspace/opt_125m_model_sharegpt")
    parser.add_argument("--predict-tokenizer", type=str, default="/workspace/opt-125m")

    args = parser.parse_args()

    if args.tokenizer == None:
        args.tokenizer = args.model
    

    
    tokenizer = get_tokenizer(args.tokenizer)
    chunkrunner = ChunkRunner(tokenizer = tokenizer,
                              chunk_size = args.chunk_size,
                              chunk_num = args.chunk_num)
    model_name = args.model
    chunkrunner.set_self_configs(model = model_name, tensor_parallel_size = args.tp)
    
    # chunkrunner.set_predict_model_and_tokenizer(predict_tokenizer_path = args.predict_tokenizer,
    #                                         predict_model_path = args.predict_model)

    chunkrunner.set_parallel_chunkworkers()
    
    
    #small model
    chunkrunner_125m = ChunkRunner(tokenizer = None,
                              chunk_size = 512,
                              chunk_num = 10)
    
    chunkrunner_125m.set_predict_model_and_tokenizer(predict_tokenizer_path = args.predict_tokenizer,
                                                     predict_model_path = args.predict_model)
    
    thread_13b = threading.Thread(target = execute_13b_model)
    thread_125m = threading.Thread(target = execute_125m_model)
    thread_13b.start()
    thread_125m.start()
    
    
    # mmap_warm()

    #engine_args = AsyncEngineArgs.from_cli_args(args)
    #engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
