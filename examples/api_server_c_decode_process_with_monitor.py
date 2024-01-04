import argparse
import asyncio
import multiprocessing
import threading
import time
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import mmap
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

app = FastAPI()

TIMEOUT_KEEP_ALIVE = 5  # seconds.
decode_event = multiprocessing.Event()


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

class SharedEngine:
    def __init__(self, engine_args: AsyncEngineArgs):
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)

    @property
    def engine(self) -> AsyncLLMEngine:
        return self._engine

def init_mdecode_prefill(shared_engine: SharedEngine, shared_string):
    # Your implementation here
    mdecode_status = shared_string
    engine = shared_engine.engine
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

def notify_mdecode(shared_engine: SharedEngine, shared_string):
    # Your implementation here
    # engine = shared_engine.engine
    # global decode_event
    # global mdecode_status
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
            shared_engine.engine.convert_req_label_status(request_id, label)
            add_time = time.time()
            print("decode get data " , request_id, arrive_time, add_time, add_time-arrive_time)

            # mdecode_status = "decode"
            shared_string = "decode"
            already_num = already_num + 1
            decode_event.set()

@app.post("/init_mdecode")
async def init_mdecode(request: Request):
    # Your implementation here
    # engine = shared_engine.engine
    request_dict = await request.json()

    request_ids = request_dict.pop("request_ids")
    prompts = request_dict.pop("prompts")
    output_lens = request_dict.pop("output_lens")
    stream = request_dict.pop("stream", False)
    sampling_params_list = []
    for i in range(len(prompts)):
        sampling_params = SamplingParams(**request_dict)
        sampling_params_list.append(sampling_params)
    shared_engine.engine.add_request(prompts=prompts, output_lens=output_lens, request_ids=request_ids, sampling_params=sampling_params_list)
    # Your logic here
    decode_event.set()
    return JSONResponse({"text": "test"})

def run_uvicorn(host: str, port: int):
    uvicorn.run(app, host=host, port=port, log_level="debug", timeout_keep_alive=TIMEOUT_KEEP_ALIVE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)

    
    with multiprocessing.Manager() as manager:
        shared_engine = manager.Namespace()
        parser = AsyncEngineArgs.add_cli_args(parser)
        args = parser.parse_args()

        engine_args = AsyncEngineArgs.from_cli_args(args)
        shared_engine.engine = AsyncLLMEngine.from_engine_args(engine_args)
        shared_string = manager.Value('c', "init_mdecode_prefill")

        multiprocessing.Process(target=init_mdecode_prefill, args=(shared_engine, shared_string), daemon=True).start()
        multiprocessing.Process(target=run_uvicorn, args=(args.host, args.port), daemon=True).start()
        multiprocessing.Process(target=notify_mdecode, args=(shared_engine, shared_string), daemon=True).start()
        # multiprocessing.Process(target=monitor_mdecode_info, args=(shared_engine, args.host, args.port), daemon=True).start()

        time.sleep(10)  # Allow some time for processes to run
