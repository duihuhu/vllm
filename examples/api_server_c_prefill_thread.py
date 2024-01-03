import argparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import uvicorn

from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.chunked.chunkrunner import ChunkRunner
from vllm.chunked.chunk import ChunkSamplingParams

from typing import List
import threading
import time
import mmap
import os
# import multiprocessing
# manager = multiprocessing.Manager()

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()

mp_dp = 'mprefill_to_mdispatcher.txt'

if not os.path.isfile(mp_dp):
    # create initial file
    with open(mp_dp, "w+b") as fd:
        fd.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')

mp_md_dp = 'mprefill_mdispatcher_to_mdecode_mdispatcher.txt'

if not os.path.isfile(mp_md_dp):
    # create initial file
    with open(mp_md_dp, "w+b") as fd:
        fd.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')

prefill_event = threading.Event()

def mprefill_exec_prefill():
    with open(mp_dp, "r+b") as fd:
        mm = mmap.mmap(fd.fileno(), 8, access=mmap.ACCESS_WRITE, offset=0)
        prefill_nums = 0 
        while True:
            prefill_event.wait() 
            #prefill_nums = engine.mprefill_generate_prefill(mm, prefill_nums)
            prefill_nums = chunkrunner.mprefill_generate_prefill(mm, prefill_nums)
            prefill_event.clear()
            prefill_event.wait()     
        mm.close()
        fd.close()
        
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
    for prompt in prompts:
        prompt_token_ids = tokenizer(prompt).input_ids
        prompts_token_ids_s.append(prompt_token_ids) 
    sampling_params_s: List[sampling_params] = []
    for _ in range(len(prompts_token_ids_s)):
        sampling_params = ChunkSamplingParams(temperature = temperature, top_p = 1.0, top_k = -1)
        sampling_params_s.append(sampling_params)
    
    #sampling_params_list = []
    #for i in range(len(prompts)):
    #    sampling_params = SamplingParams(**request_dict)
    #    sampling_params_list.append(sampling_params)
    # engine.add_request(prompts=prompts, output_lens=output_lens, request_ids=request_ids, sampling_params=sampling_params_list)
    #engine.add_mprefill_request(prompts=prompts, output_lens=output_lens, request_ids=request_ids, sampling_params=sampling_params_list)
    chunkrunner.add_requests_to_job_sequences(prompts_s = prompts,
                                              prompt_token_ids_s = prompts_token_ids_s, 
                                              sampling_params_s = sampling_params_s,
                                              big_size = 16,
                                              small_size = 16)
    prefill_event.set()

@app.post("/mprefill_add")
async def mprefill_add(request: Request) -> Response:
    request_dict = await request.json()
    await mprefill_add_prefill(request_dict)
    ret = {"text": 'test'}
    return JSONResponse(ret)

@app.on_event("startup")
def startup_decode_event():
    threading.Thread(target=mprefill_exec_prefill, daemon=True).start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--model", type=str, default="/workspace/models/facebook/opt-125m")
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-num", type=int, default=30)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--predict-model", type=str, default="/workspace/opt_125m_model_sharegpt")
    parser.add_argument("--predict-tokenizer", type=str, default="/workspace/models/facebook/opt-125m")
    args = parser.parse_args()

    if args.tokenizer == None:
        args.tokenizer = args.model
    
    tokenizer = get_tokenizer(args.tokenizer)
    chunkrunner = ChunkRunner(tokenizer = tokenizer,
                              chunk_size = args.chunk_size,
                              chunk_num = args.chunk_num)
    model_name = args.model
    chunkrunner.set_self_configs(model = model_name, 
                                 predict_model = args.predict_model,
                                 tensor_parallel_size = args.tp)
    #chunkrunner.set_predict_model_and_tokenizer(predict_tokenizer_path = args.predict_tokenizer,
    #                                            predict_model_path = args.predict_model)
    chunkrunner.set_parallel_chunkworkers()
    
    print("warm up...")
    chunkrunner.run_worker()
    print("end warm up")

    #engine_args = AsyncEngineArgs.from_cli_args(args)
    #engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
