from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.entrypoints.comm import EngineType, CommEngine, CommData, CommonHeader, CacheMeta, QueryMeta, QueryCacheMeta
from vllm.entrypoints.server_meta import InferResults
import vllm.global_scheduler.entrypoints_config as cfg
import uvicorn
import random
import argparse
import time
import json
from vllm.outputs import KvPreparedResponse, LayerKvPreparedResponse, VLLMLoadInfo, RequestOutput, CompletionOutput
from vllm.sequence import Logprob
import aiohttp
from vllm.entrypoints.server_meta import PrefilledMeta

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=60)
TIMEOUT_KEEP_ALIVE = 5
ITMEOUTOUT_TO_PREVENT_DEADLOCK = 1
app =FastAPI()
server=None

@app.post("/create_comm")
async def create_comm(request: Request) -> Response:
    payload = await request.json()
    dst_channel = payload.pop("dst_channel")
    nccl_id =  payload.pop("nccl_id")
    worker_type = payload.pop("worker_type")
    await server.engine.create_comm(nccl_id, dst_channel, worker_type)
    
@app.post("/get_nccl_id")
async def get_nccl_id(request: Request) -> Response:
    payload = await request.json()
    dst_channel = payload.pop("dst_channel")
    worker_type = payload.pop("worker_type")
    nccl_ids = await server.engine.get_nccl_id(dst_channel, worker_type)
    return nccl_ids

@app.post("/query_layer_kv_blocks")
async def query_layer_kv_blocks(response: Request) -> None:
    payload = await response.json()    
    merge_request_id, merge_num_blocks, current_transfer_tag, merge_is_allocated = await server.engine.prepare_layer_kv_blocks(payload)
    print("rrrrr")
    return LayerKvPreparedResponse(merge_request_id, merge_num_blocks, server.global_ranks, current_transfer_tag, merge_is_allocated).__json__()
    

@app.post("/pull_kv_cache")
async def pull_kv_cache(response: Request) -> None:
    payload = await response.json()
    query_meta = QueryMeta(**payload)
    await server.engine.pull_kv_blocks(query_meta)
    server.engine._request_tracker.new_requests_event.set()
    ret = {"ret": "success"}
    return ret

@app.post("/query_kv_cache")
async def query_kv_cache(response: Request) -> None:
    payload = await response.json()
    query_cache_meta = QueryCacheMeta(**payload)
    dcached_len = await server.engine.query_kv_blocks(query_cache_meta)
    ret = {"dcached_len": dcached_len}
    return ret

@app.post("/response_kv_result")
async def response_kv_result(response: Request) -> None:
    payload = await response.json()
    global_ranks = payload.pop("global_ranks")
    kv_response = KvPreparedResponse(**payload)
    print("response_kv_result ", kv_response.computed_blocks)
    kv_response.global_ranks = global_ranks
    await server.engine.add_kv_response(kv_response)

@app.post("/prepare_kv_result")
async def prepare_kv_result(request: Request) -> None:
    payload = await request.json()
    request_id = payload.pop("request_id")
    opp_ranks = payload.pop("opp_ranks")
    prompt_token_ids = payload.pop("prompt_token_ids")
    prompt_logprobs = payload.pop("prompt_logprobs")
    prefilled_token_id = payload.pop("prefilled_token_id")
    output_logprobs = payload.pop("output_logprobs")
    cumulative_logprob  = payload.pop("cumulative_logprob")
    sampling_params_json = payload.pop("sampling_params")
    index = payload.pop("index")
    texts = payload.pop("texts")
    finished = payload.pop("finished")
    eprefill_host = payload.pop("eprefill_host")
    eprefill_port = payload.pop("eprefill_port")
    edecode_host = payload.pop("edecode_host")
    edecode_port = payload.pop("edecode_port")
    prompt_logprobs = pprobs_key_s2i(prompt_logprobs)
    output_logprobs = cprobs_key_s2i(output_logprobs)
    
    sampling_params = SamplingParams(**sampling_params_json)
    
    request_output = RequestOutput(
        request_id=request_id,
        prompt=None,
        outputs= [CompletionOutput(index=index,
                                   text=texts[0],
                                   token_ids=prefilled_token_id,
                                   cumulative_logprob=cumulative_logprob,
                                   logprobs=output_logprobs)],
        prompt_token_ids=prompt_token_ids,
        prompt_logprobs=prompt_logprobs,
        finished=finished,
        eprefill_host=eprefill_host,
        eprefill_port=eprefill_port,
        edecode_host=edecode_host,
        edecode_port=edecode_port
    )
    request_output.global_ranks = opp_ranks
    
    results_generator = await server.engine.add_kv_results_request(request_id=request_id,
                                            sampling_params=sampling_params, request_output=request_output)
   
    #return results to global scheduler
    async def stream_results() -> AsyncGenerator[bytes, None]:
        #response to d
        async for kv_result in results_generator:
            yield (json.dumps(kv_result.__json__()) + "\0").encode("utf-8")
            break
    return StreamingResponse(stream_results())

        
@app.post("/response_kv_prepared")
async def response_kv_prepared(response: Request) -> None:
    payload = await response.json()
    global_ranks = payload.pop("global_ranks")
    kv_response = KvPreparedResponse(**payload)
    kv_response.global_ranks = global_ranks
    await server.engine.add_kv_response(kv_response)

def pprobs_key_s2i(prompt_logprobs):
    t_prompt_logprobs = []
    for logprob in prompt_logprobs:
        if logprob != None:
            t_logprob = {}
            for key, value in logprob.items():
                t_logprob[int(key)] = Logprob(**value)
        else:
            t_prompt_logprobs.append(logprob)
    return t_prompt_logprobs

def cprobs_key_s2i(cumulative_logprob):
    t_cumulative_logprob = []
    for logprob in cumulative_logprob:
        t_logprob = {}
        for key, value in logprob.items():
            t_logprob[int(key)] = Logprob(**value)
            t_cumulative_logprob.append(t_logprob)
    return t_cumulative_logprob
    
@app.post("/generate_decode")
async def generate_decode(request: Request) -> Response:
    payload = await request.json()
    start_time = time.time()
    is_layer = payload.pop("is_layer")
    if not is_layer:
        request_id = payload.pop("request_id")
        opp_ranks = payload.pop("opp_ranks")
        prompt_token_ids = payload.pop("prompt_token_ids")
        prompt_logprobs = payload.pop("prompt_logprobs")
        prefilled_token_id = payload.pop("prefilled_token_id")
        output_logprobs = payload.pop("output_logprobs")
        cumulative_logprob  = payload.pop("cumulative_logprob")
        sampling_params_json = payload.pop("sampling_params")
        index = payload.pop("index")
        texts = payload.pop("texts")
        finished = payload.pop("finished")
        eprefill_host = payload.pop("eprefill_host")
        eprefill_port = payload.pop("eprefill_port")
        edecode_host = payload.pop("edecode_host")
        edecode_port = payload.pop("edecode_port")
        # print("request_id ", request_id)
        
        prompt_logprobs = pprobs_key_s2i(prompt_logprobs)
        output_logprobs = cprobs_key_s2i(output_logprobs)
        
        
        sampling_params = SamplingParams(**sampling_params_json)
        
        request_output = RequestOutput(
            request_id=request_id,
            prompt=None,
            outputs= [CompletionOutput(index=index,
                                    text=texts[0],
                                    token_ids=prefilled_token_id,
                                    cumulative_logprob=cumulative_logprob,
                                    logprobs=output_logprobs)],
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=prompt_logprobs,
            finished=finished,
            eprefill_host=eprefill_host,
            eprefill_port=eprefill_port,
            edecode_host=edecode_host,
            edecode_port=edecode_port
        )
        request_output.global_ranks = opp_ranks
        results_generator = server.engine.generate(None, sampling_params=sampling_params, request_id=request_id,
                                                prompt_token_ids=prompt_token_ids, prefill_request_output=request_output, is_layer=is_layer)
    else:
        request_id = payload.pop("request_id")
        prefilled_token_id = payload.pop("prefilled_token_id")
        output_logprobs = payload.pop("output_logprobs")
        sampling_params_json = payload.pop("sampling_params")
        sampling_params = SamplingParams(**sampling_params_json)
        output_logprobs = cprobs_key_s2i(output_logprobs)
        results_generator = server.engine.generate(request_id=request_id, prefilled_token_id=prefilled_token_id, output_logprobs=output_logprobs, is_layer=is_layer)
    #return results to global scheduler
    async def stream_results() -> AsyncGenerator[bytes, None]:
        last_time = start_time
        #response to p
        # if not server.engine.engine.deploy_config.enable_layer:
        if not is_layer:
            async for kv_response in results_generator:
                yield (json.dumps(kv_response.__json__()) + "\0").encode("utf-8")
                break
        
        #response to decode
        async for request_output in results_generator:
            end_time = time.time()
            infer_result = InferResults(
                request_id = request_output.request_id,
                opp_ranks = request_output.global_ranks,
                prompt_token_ids = request_output.prompt_token_ids,
                prompt_logprobs = request_output.prompt_logprobs,
                prefilled_token_id = request_output.outputs[0].token_ids,
                output_logprobs = request_output.outputs[0].logprobs,
                cumulative_logprob = request_output.outputs[0].cumulative_logprob,
                sampling_params = sampling_params,
                index = request_output.outputs[0].index,
                texts = [output.text for output in request_output.outputs],
                finished = request_output.finished,
                jct = end_time - start_time,
                tbt = end_time - last_time,
                n = -1,
                start_time=start_time,
                end_time=end_time,
                eprefill_host = request_output.eprefill_host,
                eprefill_port = request_output.eprefill_port,
                edecode_host = request_output.edecode_host,
                edecode_port = request_output.edecode_port
            )
            last_time = end_time
            if server.engine.engine.deploy_config.enable_dcache and infer_result.finished==True:
                prefill_kv_resp = asyc_forward_request(infer_result.__json__(), cfg.forward_eprefill_res_kv_url % 
                                                                    (infer_result.eprefill_host, infer_result.eprefill_port))
                async for resp in prefill_kv_resp:
                    resp = resp.decode('utf-8')
                    payload = json.loads(resp)
                    global_ranks = payload.pop("global_ranks")
                    kv_response = KvPreparedResponse(**payload)
                    kv_response.global_ranks = global_ranks
                    # print("decode finished payload ", payload)
                    await server.engine.add_kv_response(kv_response)
                    
            yield (json.dumps(infer_result.__json__()) + "\0").encode("utf-8")
    
    return StreamingResponse(stream_results())
    

async def asyc_forward_request(request_dict, api_url):
    headers = {"User-Agent": "Test Client"}
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(url=api_url, json=request_dict,
                                    headers=headers) as response:
                if response.status == 200:
                    delimiter=b"\0"
                    buffer = b''  # 用于缓存数据块中的部分消息
                    async for chunk in response.content.iter_any():
                        buffer += chunk  # 将新的数据块添加到缓冲区中
                        while delimiter in buffer:
                            index = buffer.index(delimiter)  # 查找分隔符在缓冲区中的位置
                            message = buffer[:index]  # 提取从缓冲区起始位置到分隔符位置的消息
                            yield message.strip()  # 返回提取的消息
                            buffer = buffer[index + len(delimiter):]  # 从缓冲区中移除已提取的消息和分隔符
                else:
                    print(f"Failed response for request {response.status}")
    except aiohttp.ClientError as e:
         print(f"Request {request_dict['request_id']} failed: {e}")
    except Exception as e:
         print(f"Unexpected error for request {request_dict['request_id']}: {e}")
        
@app.post("/generate_prefill")
async def generate_prefill(request: Request) -> Response:
    """Generate completion for request

    Args:
        request (Request): _description_

    Returns:
        Response: _description_
    """
    payload = await request.json()

    # prompt = payload.pop("prompt")
    start_time = time.time()
    stream = payload.pop("stream")
    prompt_token_ids = payload.pop("prompt_token_ids")
    request_id = payload.pop("request_id")
    eprefill_host = payload.pop("eprefill_host")
    eprefill_port = payload.pop("eprefill_port")
    edecode_host = payload.pop("edecode_host")
    edecode_port = payload.pop("edecode_port")
    
    if args.enable_breakdown:
        with open("prefill_request_in.txt", "a+") as fd:
            content = "prefill request in " + request_id + " " + str(time.time())
            fd.write(content + "\n")
    cache_meta = None
    if "cmeta_host" in payload:
        cmeta_host =  payload.pop("cmeta_host")
        cmeta_port =  payload.pop("cmeta_port")
        cmeta_ranks =  payload.pop("cmeta_ranks")
        cmeta_kv_len = payload.pop("cmeta_kv_len")
        cache_meta = CacheMeta(cmeta_host, cmeta_port, cmeta_ranks, cmeta_kv_len)
        print("matched decode instance " ,cmeta_host, cmeta_port, cmeta_ranks)
    #todo 适配prefix_req 结合本地缓存复用策略
    sampling_params = SamplingParams(**payload)
    results_generator = server.engine.generate(prompt=None, prompt_token_ids=prompt_token_ids, \
        sampling_params=sampling_params, request_id=request_id, cache_meta=cache_meta, eprefill_host=eprefill_host, eprefill_port=eprefill_port, edecode_host=edecode_host, edecode_port=edecode_port)
    #Streaming case
    n = 0 
    async def stream_results() -> AsyncGenerator[bytes, None]:
        nonlocal n
        last_time = start_time
        async for request_output in results_generator:
            end_time = time.time()
            # print("request_output " , request_output)
            infer_results = InferResults(
                request_id = request_output.request_id,
                opp_ranks = request_output.global_ranks,
                prompt_token_ids = request_output.prompt_token_ids,
                prompt_logprobs = request_output.prompt_logprobs,
                prefilled_token_id = request_output.outputs[0].token_ids,
                output_logprobs = request_output.outputs[0].logprobs,
                cumulative_logprob = request_output.outputs[0].cumulative_logprob,
                sampling_params = sampling_params,
                index = request_output.outputs[0].index,
                texts = [output.text for output in request_output.outputs],
                finished = request_output.finished,
                ttft = end_time-last_time,
                jct =  end_time-last_time,
                tbt =  end_time-last_time,
                n = n,
                start_time=start_time,
                end_time=end_time,
                is_layer = request_output.is_layer,
                eprefill_host = request_output.eprefill_host,
                eprefill_port = request_output.eprefill_port,
                edecode_host = request_output.edecode_host,
                edecode_port = request_output.edecode_port
            )
            if not args.enable_separate:
                yield (json.dumps(infer_results.__json__()) + "\0").encode("utf-8")
                last_time = end_time
                n = n + 1
            else:
                layer_infer_results = PrefilledMeta(
                    request_id = request_output.request_id,
                    output_logprobs = request_output.outputs[0].logprobs,
                    prefilled_token_id = request_output.outputs[0].token_ids,
                    sampling_params = sampling_params,
                    is_layer = request_output.is_layer
                )
            
                if n==0:
                    yield (json.dumps(infer_results.__json__()) + "\0").encode("utf-8")
                last_time = end_time
                n = n + 1
        
                #send kv allocate to decode directly
                if args.enable_direct and not request_output.is_layer:
                    if infer_results.finished != True:
                        if args.enable_breakdown:
                            with open("prefill_send_query_kv_to_decode.txt", "a+") as fd:
                                content = "prefill send query kv to decode " + infer_results.request_id + " " + str(time.time())
                                fd.write(content + "\n")
                        # port = random.choice([cfg.edecode_port, cfg.edecode_port1])
                        decode_response = asyc_forward_request(infer_results.__json__(), cfg.forward_edecode_url % 
                                                                    (infer_results.edecode_host, infer_results.edecode_port))
                        d_num = 0
                else:
                    if infer_results.finished != True:
                        print("asyc_forward_request ", cfg.forward_edecode_url)
                        decode_response = asyc_forward_request(layer_infer_results.__json__(), cfg.forward_edecode_url % 
                                                                    (infer_results.edecode_host, infer_results.edecode_port))
                        d_num = 0       
                #recv kv allocate result and deocde's decode
                if args.enable_direct: 
                    async for resp in decode_response:
                        resp = resp.decode('utf-8')
                        payload = json.loads(resp)
                        if not request_output.is_layer:
                            if d_num == 0:
                                global_ranks = payload.pop("global_ranks")
                                kv_response = KvPreparedResponse(**payload)
                                # print("response_kv_result ", kv_response.computed_blocks)
                                kv_response.global_ranks = global_ranks
                                await server.engine.add_kv_response(kv_response)
                            else:
                                yield (json.dumps(payload, ensure_ascii=False) + "\0").encode("utf-8")
                        else:
                            yield (json.dumps(payload, ensure_ascii=False) + "\0").encode("utf-8")
                        d_num = d_num + 1
    return StreamingResponse(stream_results())


class ServerArgs:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        local_host: str,
        local_port: int,
        report_interval_time = 0.5
    ) -> None:
        self.engine_args = engine_args
        self.local_host = local_host
        self.local_port = local_port
        self.report_interval_time = report_interval_time
    

class Server:
    def __init__(self, server_args: ServerArgs) -> None:
        self._init_server(server_args)
    
    def _init_server(self, server_args: ServerArgs):
        self.local_entry_point = (server_args.local_host, server_args.local_port)
        self.gs_entry_point = (cfg.global_scheduler_ip, cfg.global_scheduler_port)
        
        engine_args = server_args.engine_args
        enable_separate = engine_args.enable_separate
        
        role = engine_args.role
        if not enable_separate:
            self.engine_type = EngineType.EPD
        elif role == 'prompt':
            self.engine_type = EngineType.EPREFILL
        else:
            self.engine_type = EngineType.EDECODE
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args=engine_args)
        self.global_ranks = self.engine.engine.get_global_ranks()
        
        # self.reporter = threading.Thread(target=self.report_local_info, args=(server_args.report_interval_time,))
        # self.reporter.start()
    
    def report_local_info(self, report_interval_time: float):
        #todo 从engine中获得相关负载信息，目前手动构造
        while True:
            time.sleep(report_interval_time)
            load_info = VLLMLoadInfo(
                used_gpu_blocks=0,
                used_cpu_blocks=0,
                remained_gpu_blocks=0,
                remained_cpu_blocks=0,
                num_unfinished_requests=0,
                global_ranks = self.global_ranks,
                timestamp=time.time()
            )
            data = CommData(
                headers=CommonHeader(self.local_entry_point[0], self.local_entry_point[1],
                                     self.engine_type).__json__(),
                payload=load_info.__json__()
            )
            CommEngine.send_to(self.gs_entry_point, "monitor_report", data)
            
    def run_server(self):
        uvicorn.run(app,
                    host=self.local_entry_point[0],
                    port=self.local_entry_point[1],
                    log_level="info",
                    timeout_keep_alive=TIMEOUT_KEEP_ALIVE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_host", type=str)
    parser.add_argument("--local_port", type=int)
    parser.add_argument("--enable-direct", action="store_true")
    parser.add_argument("--enable-breakdown", action="store_true")
    parser.add_argument("--enable-spee", action="store_true")
    parser.add_argument("--enable-layer", action="store_true")
    parser.add_argument('--enable-separate', action="store_true", help=('separate or not '))
    
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)
    server_args = ServerArgs(engine_args, args.local_host, args.local_port)
    server = Server(server_args)
    server.run_server()