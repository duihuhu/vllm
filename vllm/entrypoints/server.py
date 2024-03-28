from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.entrypoints.comm import EngineType, CommEngine, CommData, CommonHeader
from vllm.entrypoints.server_meta import InferResults
import entrypoints_config as cfg
import uvicorn
import threading
import queue
import argparse
import time
import json
from vllm.outputs import KvPreparedResponse, VLLMLoadInfo, RequestOutput, CompletionOutput

TIMEOUT_KEEP_ALIVE = 5
ITMEOUTOUT_TO_PREVENT_DEADLOCK = 1
app =FastAPI()
server=None

@app.post("/response_kv_prepared")
async def response_kv_prepared(response: Request) -> None:
    payload = await response.json()
    request_id = payload.pop("request_id")
    kv_response = KvPreparedResponse(**payload)
    await server.engine.add_kv_response(request_id, kv_response)

@app.post("/generate_decode")
async def generate_decode(request: Request) -> Response:
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
    
    sampling_params = SamplingParams(**sampling_params_json)
    
    # #reconstuct output_logprobs
    # output_logprobs_new = [{}] * len(output_logprobs)
    # for idx, output_logprob in enumerate(output_logprobs):
    #     for output_logprob_key, output_logprob_value in output_logprob.items():
    #         output_logprobs_new[idx][int(output_logprob_key)] = output_logprob_value
    
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
    )
    request_output.global_ranks = opp_ranks
    
    results_generator = server.engine.generate(None, sampling_params=sampling_params, request_id=request_id,
                                               prompt_token_ids=prompt_token_ids, prefill_request_output=request_output)
    
    #return results to global scheduler
    async def stream_results() -> AsyncGenerator[bytes, None]:
        #response to p
        async for kv_response in results_generator:
            yield (json.dumps(kv_response.__json__()) + "\0").encode("utf-8")
            break
        
        #response to decode
        async for request_output in results_generator:
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
                finished = request_output.finished
            )
            yield (json.dumps(infer_result.__json__()) + "\0").encode("utf-8")
    
    return StreamingResponse(stream_results())
    
    
@app.post("/generate_prefill")
async def generate_prefill(request: Request) -> Response:
    """Generate completion for request

    Args:
        request (Request): _description_

    Returns:
        Response: _description_
    """
    payload = await request.json()
    prompt = payload.pop("prompt")
    request_id = payload.pop("request_id")

    #todo 适配prefix_req 结合本地缓存复用策略
    sampling_params = SamplingParams(**payload)
    results_generator = server.engine.generate(prompt, sampling_params, request_id)
    
    #Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
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
                finished = request_output.finished
            )
            print(json.dumps(infer_results.__json__()))
            yield (json.dumps(infer_results.__json__()) + "\0").encode("utf-8")

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
                    log_level="debug",
                    timeout_keep_alive=TIMEOUT_KEEP_ALIVE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_host", type=str)
    parser.add_argument("--local_port", type=int)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)
    server_args = ServerArgs(engine_args, args.local_host, args.local_port)
    server = Server(server_args)
    server.run_server()