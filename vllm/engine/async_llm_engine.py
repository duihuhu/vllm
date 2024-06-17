import asyncio
import os
import time
from functools import partial
from typing import (AsyncIterator, Callable, Dict, Iterable, List, Optional,
                    Set, Tuple, Type, Union)

from transformers import PreTrainedTokenizer
from vllm.config import ModelConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.ray_utils import initialize_ray_cluster, ray
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput, KvPreparedResponse, LayerKvPreparedResponse, MergeReqInfo, VLLMLoadInfo
from vllm.sampling_params import SamplingParams
from vllm.sequence import MultiModalData, SequenceStatus, SequenceGroup, Sequence
from vllm.usage.usage_lib import UsageContext
from vllm.entrypoints.comm import CacheMeta, CommEngine, CommData, CommonHeader, QueryMeta, QueryCacheMeta
import json
import vllm.global_scheduler.entrypoints_config as cfg
from vllm.entrypoints.server_meta import QueryLayerKvBlocks, PrefilledMeta
from vllm._C import trans_ops
from vllm.core.interfaces import AllocStatus
from vllm.utils import random_uuid
logger = init_logger(__name__)
ENGINE_ITERATION_TIMEOUT_S = int(
    os.environ.get("VLLM_ENGINE_ITERATION_TIMEOUT_S", "60"))


class AsyncEngineDeadError(RuntimeError):
    pass


def _raise_exception_on_finish(
        task: asyncio.Task, error_callback: Callable[[Exception],
                                                     None]) -> None:
    msg = ("Task finished unexpectedly. This should never happen! "
           "Please open an issue on Github.")

    exception = None
    try:
        task.result()
        # NOTE: This will be thrown if task exits normally (which it should not)
        raise AsyncEngineDeadError(msg)
    except Exception as e:
        exception = e
        logger.error("Engine background task failed", exc_info=e)
        error_callback(exception)
        raise AsyncEngineDeadError(
            msg + " See stack trace above for the actual cause.") from e


class AsyncStream:
    """A stream of RequestOutputs for a request that can be
    iterated over asynchronously."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue = asyncio.Queue()
        self._finished = False

    def put(self, item: Union[RequestOutput, Exception]) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopAsyncIteration())
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self) -> RequestOutput:
        result = await self._queue.get()
        if isinstance(result, Exception):
            raise result
        return result


class RequestTracker:
    """Synchronous abstraction for tracking requests."""

    def __init__(self, enable_layer, enable_breakdown) -> None:
        self._request_streams: Dict[str, AsyncStream] = {}
        self._finished_requests: asyncio.Queue[str] = asyncio.Queue()
        self._new_requests: asyncio.Queue[Tuple[AsyncStream,
                                                dict]] = asyncio.Queue()
        self.new_requests_event = asyncio.Event()
        
        self._kv_responses: asyncio.Queue[Tuple[AsyncStream,
                                        dict]] = asyncio.Queue()
    
        self._kv_results_requests_streams: Dict[str, AsyncStream] = {}
        self._kv_results_requests: asyncio.Queue[Tuple[AsyncStream,
                                    dict]] = asyncio.Queue()

        self.enable_layer = enable_layer
        self.enable_breakdown = enable_breakdown
    def __contains__(self, item):
        return item in self._request_streams

    def __len__(self) -> int:
        return len(self._request_streams)

    def propagate_exception(self,
                            exc: Exception,
                            request_id: Optional[str] = None) -> None:
        """Propagate an exception to request streams
        (all if request_id is None)."""
        if request_id is not None:
            self._request_streams[request_id].put(exc)
            self.abort_request(request_id)
        else:
            for rid, stream in self._request_streams.items():
                stream.put(exc)
                self.abort_request(rid)

    def process_request_output(self,
                               is_prefill: bool,
                               global_ranks: List[int],
                               request_output: RequestOutput,
                               *,
                               verbose: bool = False) -> None:
        """Process a request output from the engine."""
        request_id = request_output.request_id
        request_output.global_ranks = global_ranks
        self._request_streams[request_id].put(request_output)
        if is_prefill or request_output.finished:
            # if verbose:
                # logger.info(f"Finished request {request_id}.")
            self.abort_request(request_id)
            if request_output.finished and self.enable_breakdown:
                if self.enable_layer:
                    with open("decode_finished_reqs_layer.txt", "a+") as fd:
                        content = "decode finish req " + request_id + " " + str(time.time())
                        fd.write(content + "\n")
                else:
                    with open("decode_finished_reqs.txt", "a+") as fd:
                        content = "decode finish req " + request_id + " " + str(time.time())
                        fd.write(content + "\n")
    
    def process_request_with_layer_output(self, is_prefill: bool, request_with_layer_output):
        self._request_streams[request_with_layer_output.request_id].put(request_with_layer_output)
        if is_prefill:
            # if verbose:
                # logger.info(f"Finished request {request_id}.")
            self.abort_request(request_with_layer_output.request_id)
           
    def process_kv_response(self,
                            global_ranks: List[int],
                            kv_response: KvPreparedResponse) -> None:
        """Process a request output from the engine"""
        request_id = kv_response.request_id
        kv_response.global_ranks = global_ranks
        self._request_streams.get(request_id).put(kv_response)
        if kv_response.error !=0:
            self.abort_request(request_id)
            
    def process_kv_results(self,
                        global_ranks: List[int],
                        kv_response: KvPreparedResponse) -> None:
        """Process a request output from the engine"""
        request_id = kv_response.request_id
        kv_response.global_ranks = global_ranks
        self._kv_results_requests_streams.get(request_id).put(kv_response)
        if kv_response.error !=0:
            self.abort_request(request_id)
            
    def process_exception(self,
                          request_id: str,
                          exception: Exception,
                          *,
                          verbose: bool = False) -> None:
        """Propagate an exception from the engine."""
        self._request_streams[request_id].put(exception)
        if verbose:
            logger.info(f"Finished request {request_id}.")
        self.abort_request(request_id)

    def add_kv_results_request(self, request_id: str,
                    **engine_add_request_kwargs) -> AsyncStream:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        if request_id in self._kv_results_requests_streams:
            raise KeyError(f"Request {request_id} already exists.")

        stream = AsyncStream(request_id)
        self._kv_results_requests.put_nowait((stream, {
            "request_id": request_id,
            **engine_add_request_kwargs
        }))
        self.new_requests_event.set()
        return stream

    def add_request(self, request_id: str,
                    **engine_add_request_kwargs) -> AsyncStream:
        """Add a request to be sent to the engine on the next background
        loop iteration."""
        if request_id in self._request_streams and self.enable_layer:
            return self._request_streams[request_id] 
        if request_id in self._request_streams:
            raise KeyError(f"Request {request_id} already exists.")
        stream = AsyncStream(request_id)
        self._new_requests.put_nowait((stream, {
            "request_id": request_id,
            **engine_add_request_kwargs
        }))
        print("add_request new_requests_event set ", request_id)
        self.new_requests_event.set()
        return stream

    def add_kv_response(self,
                        **engine_kv_response_kwargs) -> None:
        self._kv_responses.put_nowait({
            **engine_kv_response_kwargs
        })
        self.new_requests_event.set()
        
    def abort_request(self, request_id: str, *, verbose: bool = False) -> None:
        """Abort a request during next background loop iteration."""
        if verbose:
            logger.info(f"Aborted request {request_id}.")

        self._finished_requests.put_nowait(request_id)

        if request_id not in self._request_streams or self._request_streams[
                request_id].finished:
            # The request has already finished or been aborted.
            return

        self._request_streams[request_id].finish()

    def get_kv_responses(self) -> List[dict]:
        kv_responses: List = []
        while not self._kv_responses.empty():
            response = self._kv_responses.get_nowait()
            kv_responses.append(response)
        return kv_responses    
    
    def get_new_kv_results_request(self) -> List[Dict]:
        kv_results_requests: List[Dict] = []
        while not self._kv_results_requests.empty():
            stream, kv_results_request = self._kv_results_requests.get_nowait()
            self._kv_results_requests_streams[stream.request_id] = stream
            kv_results_requests.append(kv_results_request)
        return kv_results_requests
    
    def get_new_and_finished_requests(self) -> Tuple[List[Dict], Set[str]]:
        """Get the new requests and finished requests to be
        sent to the engine."""
        new_requests: List[Dict] = []
        finished_requests: Set[str] = set()

        while not self._finished_requests.empty():
            request_id = self._finished_requests.get_nowait()
            finished_requests.add(request_id)
            self._request_streams.pop(request_id, None)

        while not self._new_requests.empty():
            stream, new_request = self._new_requests.get_nowait()
            if stream.request_id in finished_requests:
                # The request has already been aborted.
                stream.finish()
                continue
            self._request_streams[stream.request_id] = stream
            new_requests.append(new_request)
            
        return new_requests, finished_requests

    async def wait_for_new_requests(self):
        if not self.has_new_requests():
            await self.new_requests_event.wait()
        self.new_requests_event.clear()

    def has_new_requests(self):
        return not self._new_requests.empty()


class _AsyncLLMEngine(LLMEngine):
    """Extension of LLMEngine to add async methods."""
    async def _pull_cache_signal(self, cache_meta, request_id, prompt_token_ids):
        decode_entry_point = (cache_meta.cmeta_host, cache_meta.cmeta_port)
        query_meta = QueryMeta(cache_meta, self.deploy_config.deploy_host, self.deploy_config.deploy_port, 
                               self.deploy_config.get_global_ranks(), request_id, prompt_token_ids).__json__()
        data = CommData(
            headers=CommonHeader(self.deploy_config.deploy_host, self.deploy_config.deploy_port).__json__(),
            payload=query_meta
        )
        return await CommEngine.async_send_to(decode_entry_point, "pull_kv_cache", data)
    
    async def _query_cache_meta(self, cache_meta: CacheMeta, request_id, prompt_token_ids):
        decode_entry_point = (cache_meta.cmeta_host, cache_meta.cmeta_port)
        query_cache_meta = QueryCacheMeta(request_id, prompt_token_ids).__json__()
        data = CommData(
            headers=CommonHeader(self.deploy_config.deploy_host, self.deploy_config.deploy_port).__json__(),
            payload=query_cache_meta
        )
        return await CommEngine.async_send_to(decode_entry_point, "query_kv_cache", data)
        
    async def _query_cache(self, seq_group: SequenceGroup, request_tracker: RequestTracker):
        seq = seq_group.get_seqs()[0] 
        query_response = await self._query_cache_meta(seq_group.cache_meta, seq_group.request_id, seq.data.prompt_token_ids)
        query_response = json.loads(query_response)
        resp_cached_len = query_response["dcached_len"]
        seq_group.cache_meta.cmeta_kv_len = resp_cached_len
        block_table = self.scheduler.block_manager.block_tables[seq.seq_id]
        phy_blocks = [phy_block for phy_block in block_table]              
        computed_blocks = [phy_block.block_number for phy_block in phy_blocks if phy_block.computed == True]
        # print("add_recv_transfering, computed_blocks, phy_blocks, dcached_len " , 
        #         len(computed_blocks), len(phy_blocks), resp_cached_len, seq_group.cache_meta.cached_len)
        
        if len(computed_blocks) == resp_cached_len:
            seq_group.cache_meta.is_ready = True
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq.status = SequenceStatus.WAITING
            self.scheduler.waiting.append(seq_group)
        else:
            self.scheduler.add_recv_transfering(seq_group)
            phy_blocks_num = [phy_block.block_number for phy_block in phy_blocks]
            self.recv_kv_trans_scheduler.add_kv_request(seq_group.request_id, seq_group.cache_meta.cmeta_ranks, 
                                                    phy_blocks_num[len(computed_blocks
                                                                       ): resp_cached_len], False)
            pull_response = await self._pull_cache_signal(seq_group.cache_meta, seq_group.request_id, seq_group.prompt_token_ids)
            request_tracker.new_requests_event.set()

            print("pull_response ", pull_response)
            
    def check_deocde_recv_meta(self):
        meta_recv_finished_id = []
        for request_id, seq_group in self.scheduler.meta_recv_finished.items():
            if request_id in self.scheduler.decode_recv_finished:
                # print("check_deocde_recv_meta running " , time.time())
                self.scheduler.running.append(seq_group)
                self.scheduler.block_manager.move_kv_blocks_meta(seq_group)
                meta_recv_finished_id.append(request_id)
        for request_id in meta_recv_finished_id:
            del self.scheduler.meta_recv_finished[request_id]
            del self.scheduler.decode_recv_finished[request_id]
 

    async def _query_layer_kv_blocks(self, seq_groups: List[SequenceGroup]):
            decode_entry_point = (seq_groups[0].edecode_host, seq_groups[0].edecode_port)
            query_blocks = []
            #gather all seq_groups
            for seq_group in seq_groups:
                query_layer_block =  QueryLayerKvBlocks(seq_group.request_id, seq_group.prompt_token_ids, seq_group.sampling_params, self.get_global_ranks(), seq_group.eprefill_host, seq_group.eprefill_port, seq_group.edecode_host, seq_group.edecode_port).__json__()
                query_blocks.append(query_layer_block)
                
            data = CommData(
                headers=CommonHeader(self.deploy_config.deploy_host, self.deploy_config.deploy_port).__json__(),
                payload=query_blocks
            )
            return await CommEngine.async_send_to(decode_entry_point, "query_layer_kv_blocks", data)
        
    #get block num and global ranks
    async def query_layer_kv_blocks(self, request_tracker: RequestTracker):        
        categorized_groups: Dict[str, List[SequenceGroup]] = {}

        for seq_group in  self.scheduler.running:
            key = seq_group.edecode_host + "_" + str(seq_group.edecode_port)
            if key in categorized_groups:   
                categorized_groups[key].append(seq_group)
            else:
                categorized_groups[key] = [seq_group]
            if self.deploy_config.enable_layer and self.deploy_config.enable_breakdown:
                with open("prefill_send_query_kv_to_decode_layer.txt", "a+") as fd:
                    content = "prefill send query kv to decode " + seq_group.request_id + " " + str(time.time())
                    fd.write(content + "\n")
                    
        order_kv_request_ids = []
        order_no_kv_request_ids = []
        send_seq_groups: List[List[SequenceGroup]] = []
        coroutines = []
        for dest, seq_groups in categorized_groups.items():
            coroutines.append(asyncio.create_task(self._query_layer_kv_blocks(seq_groups)))
            send_seq_groups.append(seq_groups)
        layer_kv_responses = await asyncio.gather(*coroutines)
        merage_reqs = []
        for layer_kv_response, send_seq_group in zip(layer_kv_responses, send_seq_groups):
            layer_kv = LayerKvPreparedResponse(**layer_kv_response)
            send_blocks = []
            merge_seq_groups = []
            for seq_group, computed_blocks, is_allocated in zip(send_seq_group, layer_kv.computed_blocks, layer_kv.is_allocated):
                if is_allocated:
                    blocks = self.scheduler.fetch_kv_blocks(seq_group)
                    if computed_blocks <= len(blocks):
                        send_blocks.extend(blocks[computed_blocks:])
                        merge_seq_groups.append(seq_group)
                        self.scheduler.seq_groups_with_layer[seq_group.request_id] = seq_group
                        order_kv_request_ids.append(seq_group.request_id)
                else:
                    order_no_kv_request_ids.append(seq_group.request_id)
                    
            self.scheduler.send_transfering[layer_kv.merage_request_id] = merge_seq_groups
            self.send_kv_trans_scheduler.add_layer_kv_request(layer_kv.merage_request_id, layer_kv.global_ranks, send_blocks)
            opp_channel = "_".join([str(rank) for rank in layer_kv.global_ranks])
            merage_req = MergeReqInfo(layer_kv.merage_request_id, send_blocks, opp_channel)
            merage_reqs.append(merage_req)
        return merage_reqs, order_kv_request_ids + order_no_kv_request_ids

            
    async def step_async(self, request_tracker) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.
        The workers are ran asynchronously if possible.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        self.scheduler._check_tranfer_finished_req()
        
        #TODO evict block from gpu to dram in radix tree
        evicted_blocks_to_swap_out = None
        swap_id = None
        if self.scheduler.cache_config.enable_radix_caching and self.scheduler.cache_config.enable_radix_evictor:    
            # is_evict = self.scheduler.check_hbm_usage()
            # if is_evict:
            can_evicted_nodes, cpu_blocks = self.scheduler.get_evicted_blocks()
            if can_evicted_nodes:
                evicted_blocks_to_swap_out =  {evicted_node.value.physicalTokenBlock.block_number: cpu_block.block_number
                for evicted_node, cpu_block in zip(can_evicted_nodes, cpu_blocks)}
                if evicted_blocks_to_swap_out:
                    swap_id = random_uuid()
                    self.scheduler.add_swaping_out(swap_id, (can_evicted_nodes, cpu_blocks))
                    self.radix_swap_scheduler.add_swap_task(swap_id)
            self.scheduler._check_swap_finished()
            
        if self.deploy_config.enable_separate and self.deploy_config.role=="decoder":
            print("req recv " , len(self.scheduler.meta_recv_finished), len(self.scheduler.decode_recv_finished), len(self.scheduler.kv_prepared_seq_group), len(self.scheduler.recv_transfering))
        seq_group_metadata_list, scheduler_outputs, cached_seq_groups = self.scheduler.schedule()

        # if scheduler_outputs.is_empty():
        #     if self.scheduler.swapping_in or self.scheduler.swapping_out or \
        #         self.scheduler.send_transfering or self.scheduler.recv_transfering or self.scheduler.req_pull_send_transfering:
        #             logger.info("schedule empty but has swapping or kv transfering event sleep 0.5s")
        #             time.sleep(0.05)
        #     else:
        #         return []
        
        # prefill send seq's query kv cache to decode
        if self.deploy_config.enable_cache_meta and self.deploy_config.role == "prompt":
            if cached_seq_groups:
                for seq_group in cached_seq_groups:
                    asyncio.create_task(self._query_cache(seq_group, request_tracker))
    
        #use transfer kv cache by layer and by req, should enable_layer, and it use only in prompt
        merge_reqs_info = None    
        if self.deploy_config.enable_layer and self.deploy_config.role == "prompt" and seq_group_metadata_list:
            merge_reqs_info, order_request_ids = await self.query_layer_kv_blocks(request_tracker)
            #sort by merge req info's request id order to send block 
            order_request_ids_index: Dict[int, int] = {rid: index for index, rid in enumerate(order_request_ids)}
            seq_group_metadata_list.sort(key=lambda x: order_request_ids_index[x.request_id])
            scheduler_outputs.scheduled_seq_groups.sort(key=lambda x: order_request_ids_index[x.seq_group.request_id])

        if not scheduler_outputs.is_empty():
            # Execute the model.
            all_outputs = await self.model_executor.execute_model_async(
                seq_group_metadata_list = seq_group_metadata_list, 
                blocks_to_swap_in = scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out = scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy = scheduler_outputs.blocks_to_copy,
                merge_reqs_info = merge_reqs_info,
                evicted_blocks_to_swap_out = evicted_blocks_to_swap_out,
                swap_id=swap_id)

            self.scheduler.swap_finished_req_ids = [out[1] for out in all_outputs]
            # Only the driver worker returns the sampling results.
            output = all_outputs[0][0]
            evicted_blocks_to_swap_out = None
        else:
            output = []

        if self.scheduler.cache_config.enable_radix_caching and self.scheduler.cache_config.enable_radix_evictor and evicted_blocks_to_swap_out:
            all_outputs = await self.model_executor._run_workers_async(
                "evict_blocks",
                evicted_blocks_to_swap_out = evicted_blocks_to_swap_out,
                swap_id=swap_id)
                
        processed_outputs = self._process_model_outputs(output, scheduler_outputs)
        
        processed_output_without_layer = []
        for processed_output in processed_outputs:
            if processed_output.request_id not in self.scheduler.seq_groups_with_layer:
                processed_output_without_layer.append(processed_output)
            else:
                self.scheduler.outputs_with_layer[processed_output.request_id] = processed_output
         
        if self.deploy_config.enable_separate and self.deploy_config.role=="decoder":
            print("after _process_model_outputs req recv " , len(self.scheduler.meta_recv_finished), len(self.scheduler.decode_recv_finished), len(self.scheduler.kv_prepared_seq_group), len(self.scheduler.recv_transfering))
        #prompt eng pull metadata in separate mode
        #assume after do prefill, the reqeust will not finish
        if not self.deploy_config.enable_layer:
            if self.deploy_config.enable_separate and self.deploy_config.role == 'prompt':
                prefilled_seq_groups = self.scheduler.fetch_prefilled_seq_groups()
                for seq_group in prefilled_seq_groups:
                    self.scheduler.add_send_transfering(seq_group)
                #if enable_radix_cacheing and in separate model, we should update it when prefilled prompt
                if self.deploy_config.enable_radix_caching:
                    self.scheduler.radix_manager_update(prefilled_seq_groups)
                    
            if self.deploy_config.enable_separate and self.deploy_config.role == 'decoder' and self.deploy_config.enable_dcache:
                decoded_seq_groups = self.scheduler.fetch_decoded_seq_groups()
                for seq_group in decoded_seq_groups:
                    self.scheduler.add_send_transfering(seq_group)
        else:
            processed_output_with_layer = []
            if self.deploy_config.enable_separate and self.deploy_config.role == "prompt":
                prefilled_seq_groups = self.scheduler.fetch_prefilled_seq_groups()
                #if enable_radix_cacheing and in separate model, we should update it when prefilled prompt
                if self.deploy_config.enable_radix_caching:
                    self.scheduler.radix_manager_update(prefilled_seq_groups)
                    
                for seq_group in prefilled_seq_groups:
                    output = self.scheduler.outputs_with_layer[seq_group.request_id]
                    output.is_layer = True
                    processed_output_with_layer.append(output)
                    del self.scheduler.outputs_with_layer[seq_group.request_id]
                    del self.scheduler.seq_groups_with_layer[seq_group.request_id]
                    
            if self.deploy_config.enable_separate and self.deploy_config.role == 'decoder' and self.deploy_config.enable_dcache:
                decoded_seq_groups = self.scheduler.fetch_decoded_seq_groups()
                for seq_group in decoded_seq_groups:
                    self.scheduler.add_send_transfering(seq_group)
                    
        if self.deploy_config.enable_layer:
            return processed_output_without_layer, processed_output_with_layer
        return processed_output_without_layer, []

    async def encode_request_async(
        self,
        request_id: str,  # pylint: disable=unused-argument
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]] = None,
        lora_request: Optional[LoRARequest] = None,
    ):
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = await self.tokenizer.encode_async(
                request_id=request_id,
                prompt=prompt,
                lora_request=lora_request)
        return prompt_token_ids

    async def add_request_async(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
        prefill_request_output: Optional[RequestOutput] = None,
        cache_meta: Optional[CacheMeta] = None,
        eprefill_host: Optional[str] = None,
        eprefill_port: Optional[str] = None,
        edecode_host: Optional[str] = None,
        edecode_port: Optional[str] = None,
        prefilled_token_id: Optional[List[int]] = None,
        output_logprobs: Optional[Dict[int, float]] = None,
        is_layer: Optional[bool] = False
    ) -> None:
        if not is_layer:
            if lora_request is not None and not self.lora_config:
                raise ValueError(f"Got lora_request {lora_request} but LoRA is "
                                "not enabled!")
            if arrival_time is None:
                arrival_time = time.time()
            prompt_token_ids = await self.encode_request_async(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                lora_request=lora_request)

        return self.add_request(request_id,
                                prompt=prompt,
                                prompt_token_ids=prompt_token_ids,
                                sampling_params=sampling_params,
                                arrival_time=arrival_time,
                                lora_request=lora_request,
                                multi_modal_data=multi_modal_data,
                                prefill_request_output=prefill_request_output,
                                cache_meta=cache_meta,
                                eprefill_host=eprefill_host,
                                eprefill_port=eprefill_port,
                                edecode_host=edecode_host,
                                edecode_port=edecode_port,
                                prefilled_token_id=prefilled_token_id,
                                output_logprobs=output_logprobs,
                                is_layer=is_layer)

    async def check_health_async(self) -> None:
        self.model_executor.check_health()

    async def trans_kv_step_aysnc(self) -> None:
        if not self.deploy_config.enable_separate:
            return
        if not self.scheduler.send_transfering and not self.scheduler.recv_transfering and not self.scheduler.req_pull_send_transfering:
            return 
        if self.deploy_config.enable_debug:
            t1 = time.time()
        # print("trans_kv_step_aysnc ")
        finished_work_tasks = await self.model_executor._run_workers_async(
            "get_finished_transfer_tasks",
            # get_all_outputs=True
        )
        if self.deploy_config.enable_debug:
            t2 = time.time()   
        for finished_tasks in finished_work_tasks:
            for worker_finished_tasks in finished_tasks:
                if worker_finished_tasks:
                    for worker_finished_task in worker_finished_tasks:
                        # print("worker_finished_tasks ", finished_tasks, worker_finished_tasks)
                        send_finished_tasks = [] 
                        recv_finished_tasks = []
                        for finished_task in worker_finished_task[0]:
                            send_finished_tasks.append(trans_ops.TransferTaskMeta.deserialize(finished_task))
                        for finished_task in worker_finished_task[1]:
                            recv_finished_tasks.append(trans_ops.TransferTaskMeta.deserialize(finished_task))
                        # print("send_finished_tasks, recv_finished_tasks ", send_finished_tasks, recv_finished_tasks)
                        real_send_finished_req_ids = self.send_kv_trans_scheduler.add_finished_tasks(send_finished_tasks)
                        real_recv_finished_req_ids = self.recv_kv_trans_scheduler.add_finished_tasks(recv_finished_tasks)
                        if real_send_finished_req_ids:
                            self.scheduler.add_send_finished(real_send_finished_req_ids)
                        if real_recv_finished_req_ids:
                            self.scheduler.add_recv_finished(real_recv_finished_req_ids)

        send_tasks = self.send_kv_trans_scheduler.schedule()
        recv_tasks = self.recv_kv_trans_scheduler.schedule()
        if self.deploy_config.enable_debug:
            t3 = time.time()   
        if send_tasks or recv_tasks:
            await self.model_executor._run_workers_async(
                "trans_blocks",
                send_tasks=send_tasks,
                recv_tasks=recv_tasks
            )
        if self.deploy_config.enable_debug:
            t4 = time.time()
            self.trans_checked_time = self.trans_checked_time + t2 - t1
            self.trans_sched_time = self.trans_checked_time + t3 - t2
            self.trans_running_time = self.trans_running_time + t4 - t3
            self.trans_kv_turns  = self.trans_kv_turns + 1
            
    async def swap_step_aysnc(self) -> None:
        if not self.scheduler.radix_swapping:
            return
        
        finished_swap_tasks = await self.model_executor._run_workers_async(
            "get_finished_swap_tasks",
        )
        # print("finished_swap_tasks ", finished_swap_tasks)
        for finished_tasks in finished_swap_tasks:
            for swap_finished_task in finished_tasks:
                real_swap_finished_swap_ids = self.radix_swap_scheduler.add_finished_tasks(swap_finished_task)
                self.scheduler.add_swap_out_finished(real_swap_finished_swap_ids)
                
class AsyncLLMEngine:
    """An asynchronous wrapper for LLMEngine.

    This class is used to wrap the LLMEngine class to make it asynchronous. It
    uses asyncio to create a background loop that keeps processing incoming
    requests. The LLMEngine is kicked by the generate method when there
    are requests in the waiting queue. The generate method yields the outputs
    from the LLMEngine to the caller.

    NOTE: For the comprehensive list of arguments, see `LLMEngine`.

    Args:
        worker_use_ray: Whether to use Ray for model workers. Required for
            distributed execution. Should be the same as
            `parallel_config.worker_use_ray`.
        engine_use_ray: Whether to make LLMEngine a Ray actor. If so, the
            async frontend will be executed in a separate process as the
            model workers.
        log_requests: Whether to log the requests.
        max_log_len: Maximum number of prompt characters or prompt ID numbers
            being printed in log.
        start_engine_loop: If True, the background task to run the engine
            will be automatically started in the generate call.
        *args: Arguments for LLMEngine.
        *kwargs: Arguments for LLMEngine.
    """

    _engine_class: Type[_AsyncLLMEngine] = _AsyncLLMEngine

    def __init__(self,
                 worker_use_ray: bool,
                 engine_use_ray: bool,
                 *args,
                 log_requests: bool = True,
                 max_log_len: Optional[int] = None,
                 start_engine_loop: bool = True,
                 **kwargs) -> None:
        self.worker_use_ray = worker_use_ray
        self.engine_use_ray = engine_use_ray
        self.log_requests = log_requests
        self.max_log_len = max_log_len
        self.engine = self._init_engine(*args, **kwargs)

        self.background_loop = None
        # We need to keep a reference to unshielded
        # task as well to prevent it from being garbage
        # collected
        self._background_loop_unshielded = None
        self.start_engine_loop = start_engine_loop
        self._request_tracker: Optional[RequestTracker] = None
        self._errored_with: Optional[BaseException] = None
        self.transfer_time = 0
        self.engine_time = 0
        self.start_engine_time = 0
    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
    ) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        device_config = engine_configs[4]

        if device_config.device_type == "neuron":
            raise NotImplementedError("Neuron is not supported for "
                                      "async engine yet.")
        elif parallel_config.worker_use_ray or engine_args.engine_use_ray:
            initialize_ray_cluster(parallel_config)
            from vllm.executor.ray_gpu_executor import RayGPUExecutorAsync
            executor_class = RayGPUExecutorAsync
        else:
            assert parallel_config.world_size == 1, (
                "Ray is required if parallel_config.world_size > 1.")
            from vllm.executor.gpu_executor import GPUExecutorAsync
            executor_class = GPUExecutorAsync
        # Create the async LLM engine.
        engine = cls(
            parallel_config.worker_use_ray,
            engine_args.engine_use_ray,
            *engine_configs,
            executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            max_log_len=engine_args.max_log_len,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
        )
        return engine

    @property
    def is_running(self) -> bool:
        return (self.background_loop is not None
                and not self._background_loop_unshielded.done())

    @property
    def is_stopped(self) -> bool:
        return self.errored or (self.background_loop is not None
                                and self._background_loop_unshielded.done())

    @property
    def errored(self) -> bool:
        return self._errored_with is not None

    def set_errored(self, exc: Exception) -> None:
        self._errored_with = exc

    def _error_callback(self, exc: Exception) -> None:
        self.set_errored(exc)
        self._request_tracker.propagate_exception(exc)

    async def get_tokenizer(self) -> "PreTrainedTokenizer":
        if self.engine_use_ray:
            return await self.engine.get_tokenizer.remote()
        else:
            return self.engine.get_tokenizer()

    def start_background_loop(self) -> None:
        """Start the background loop."""
        if self.errored:
            raise AsyncEngineDeadError(
                "Background loop has errored already.") from self._errored_with
        if self.is_running:
            raise RuntimeError("Background loop is already running.")
        # Initialize the RequestTracker here so it uses the right event loop.
        self._request_tracker = RequestTracker(self.engine.deploy_config.enable_layer, self.engine.deploy_config.enable_breakdown)

        self._background_loop_unshielded = asyncio.get_event_loop(
        ).create_task(self.run_engine_loop())
        self._background_loop_unshielded.add_done_callback(
            partial(_raise_exception_on_finish,
                    error_callback=self._error_callback))
        self.background_loop = asyncio.shield(self._background_loop_unshielded)

    def _init_engine(self, *args,
                     **kwargs) -> Union[_AsyncLLMEngine, "ray.ObjectRef"]:
        if not self.engine_use_ray:
            engine_class = self._engine_class
        elif self.worker_use_ray:
            engine_class = ray.remote(num_cpus=0)(self._engine_class).remote
        else:
            # FIXME(woosuk): This is a bit hacky. Be careful when changing the
            # order of the arguments.
            cache_config = args[1]
            parallel_config = args[2]
            if parallel_config.tensor_parallel_size == 1:
                num_gpus = cache_config.gpu_memory_utilization
            else:
                num_gpus = 1
            engine_class = ray.remote(num_gpus=num_gpus)(
                self._engine_class).remote
        return engine_class(*args, **kwargs)

    async def engine_step(self) -> bool:
        """Kick the engine to process the waiting requests.

        Returns True if there are in-progress requests."""
        # t1 = time.time()
        new_requests, finished_requests = (
            self._request_tracker.get_new_and_finished_requests())
        for new_request in new_requests:
            # Add the request into the vLLM engine's waiting queue.
            # TODO: Maybe add add_request_batch to reduce Ray overhead
            try:
                if self.engine_use_ray:
                    await self.engine.add_request.remote(**new_request)
                else:
                    await self.engine.add_request_async(**new_request)
            except ValueError as e:
                # TODO: use a vLLM specific error for failed validation
                self._request_tracker.process_exception(
                    new_request["request_id"],
                    e,
                    verbose=self.log_requests,
                )
        if finished_requests:
            await self._engine_abort(finished_requests)

        #recv kv_responses in , sender get allocated kv cache notify from receiver(waht ever p or d, both all them use this)
        kv_responses = self._request_tracker.get_kv_responses()
        for kv_response in kv_responses:
            # Add the response
            if self.engine_use_ray:
                await self.engine.add_kv_response.remote(**kv_response)
            else:
                self.engine.add_kv_response(**kv_response)
    
        #kv_responses out, receiver process allocate kv cache req from sender, and return allocat kv num
        kv_responses = self.engine.schedule_decode_waiting()
        for kv_response in kv_responses:
            self._request_tracker.process_kv_response(
                self.engine.get_global_ranks(), kv_response)
        
        #d to p, p allocate kv cache for decode transfer data back
        kv_results_requests = self._request_tracker.get_new_kv_results_request()
        for kv_result_requests in kv_results_requests:
            kv_response = None
            if self.engine_use_ray:
                kv_response = await self.engine.add_kv_results_request.remote(**kv_result_requests)
            else:
                kv_response = self.engine.add_kv_results_request(**kv_result_requests)
            if kv_response:
                self._request_tracker.process_kv_results(
                    self.engine.get_global_ranks(), kv_response)
                
        ##trans_kv_step response for check request finished send/recv and start a send/recv task
        if self.engine_use_ray:
            await self.engine.trans_kv_step.remote()
            request_outputs = await self.engine.step.remote()
        else:
            if self.engine.deploy_config.enable_debug:
                t2 = time.time()
            await self.engine.trans_kv_step_aysnc()
            if self.engine.deploy_config.enable_debug:
                t3 = time.time()
                self.transfer_time = self.transfer_time + t3 - t2
            request_outputs, request_with_layer_outputs = await self.engine.step_async(self._request_tracker)
            if self.engine.deploy_config.enable_debug:
                t4 = time.time()
                self.engine_time = self.engine_time + t4 - t2

            if self.engine.scheduler.cache_config.enable_radix_caching and self.engine.scheduler.cache_config.enable_radix_evictor:  
                await self.engine.swap_step_aysnc()
        # Put the outputs into the corresponding streams.
        for request_output in request_outputs:
            self._request_tracker.process_request_output(
                self.engine.deploy_config.enable_separate and self.engine.deploy_config.role == "prompt",
                self.engine.get_global_ranks(),
                request_output, verbose=self.log_requests)
        
        for request_with_layer_output in request_with_layer_outputs:
            self._request_tracker.process_request_with_layer_output(
                self.engine.deploy_config.enable_separate and self.engine.deploy_config.role == "prompt",
                request_with_layer_output
            )
        
        return len(request_outputs) > 0 or len(request_with_layer_outputs) > 0 

    async def _engine_abort(self, request_ids: Iterable[str]):
        if self.engine_use_ray:
            await self.engine.abort_request.remote(request_ids)
        else:
            self.engine.abort_request(request_ids)

    async def run_engine_loop(self):
        has_requests_in_progress = False
        while True:
            if (not has_requests_in_progress and
                not self.engine.scheduler.radix_swapping and
                not self.engine.scheduler.recv_transfering and
                not self.engine.scheduler.send_transfering and
                not self.engine.scheduler.req_pull_send_transfering 
                # and
                # not self.engine.scheduler.decode_recv_finished and
                # not self.engine.scheduler.meta_recv_finished and
                # not self.engine.scheduler.kv_prepared_seq_group
                ):
                
                logger.debug("Waiting for new requests...")
                await self._request_tracker.wait_for_new_requests()
                logger.debug("Got new requests!")
            # Abort if iteration takes too long due to unrecoverable errors
            # (eg. NCCL timeouts).
            try:
                has_requests_in_progress = await asyncio.wait_for(
                    self.engine_step(), ENGINE_ITERATION_TIMEOUT_S)
                # if self.engine.deploy_config.enable_debug:
                #     if (not has_requests_in_progress and
                #         not self.engine.scheduler.swapping_in and
                #         not self.engine.scheduler.swapping_out and
                #         not self.engine.scheduler.recv_transfering and
                #         not self.engine.scheduler.send_transfering):
                #         trans_blocks_time = await self.engine.model_executor._run_workers_async(
                #             "get_trans_blocks_time",
                #         )
                #         print("trans block time, transfer time, engine time, trans_checked_time, trans_sched_time,trans_running_time ", trans_blocks_time[0], trans_blocks_time[1], self.transfer_time, self.engine_time, self.engine.trans_checked_time, self.engine.trans_sched_time, self.engine.trans_running_time, self.engine.trans_kv_turns)

            except asyncio.TimeoutError as exc:
                logger.error(
                    "Engine iteration timed out. This should never happen!")
                self.set_errored(exc)
                raise
            await asyncio.sleep(0)

    async def add_kv_results_request(
        self, 
        request_id: str,
        sampling_params: SamplingParams,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
        request_output: Optional[RequestOutput] = None
    ) -> AsyncStream:
        stream = self._request_tracker.add_kv_results_request(
            request_id=request_id,
            sampling_params=sampling_params,
            lora_request = lora_request, 
            multi_modal_data = multi_modal_data,
            request_output = request_output
        )
        return stream
        
    async def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
        prefill_request_output: Optional[RequestOutput] = None,
        cache_meta: Optional[CacheMeta] = None,
        eprefill_host: Optional[str] = None,
        eprefill_port: Optional[str] = None,
        edecode_host: Optional[str] = None,
        edecode_port: Optional[str] = None,
        prefilled_token_id: Optional[List[int]] = None,
        output_logprobs: Optional[Dict[int, float]] = None,
        is_layer: Optional[bool] = False
    ) -> AsyncStream:
        if not is_layer:
            if self.log_requests:
                shortened_prompt = prompt
                shortened_token_ids = prompt_token_ids
                if self.max_log_len is not None:
                    if shortened_prompt is not None:
                        shortened_prompt = shortened_prompt[:self.max_log_len]
                    if shortened_token_ids is not None:
                        shortened_token_ids = shortened_token_ids[:self.
                                                                max_log_len]
                # logger.info(f"Received request {request_id}: "
                #             f"prompt: {shortened_prompt!r}, "
                #             f"sampling_params: {sampling_params}, "
                #             f"prompt_token_ids: {shortened_token_ids}, "
                #             f"lora_request: {lora_request}.")

            if not self.is_running:
                if self.start_engine_loop:
                    self.start_engine_time = time.time()
                    self.start_background_loop()
                else:
                    raise AsyncEngineDeadError(
                        "Background loop is not running. If it was running, "
                        "inspect the output to find the stacktrace of the "
                        "error that caused the background loop to stop "
                        "(AsyncEngineDeadError).")

            if arrival_time is None:
                arrival_time = time.time()

            if self.engine_use_ray:
                prompt_token_ids = await self.engine.encode_request_async.remote(
                    request_id=request_id,
                    prompt=prompt,
                    prompt_token_ids=prompt_token_ids,
                    lora_request=lora_request)
            else:
                prompt_token_ids = await self.engine.encode_request_async(
                    request_id=request_id,
                    prompt=prompt,
                    prompt_token_ids=prompt_token_ids,
                    lora_request=lora_request)

        stream = self._request_tracker.add_request(
            request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            arrival_time=arrival_time,
            lora_request=lora_request,
            multi_modal_data=multi_modal_data,
            prefill_request_output=prefill_request_output,
            cache_meta = cache_meta,
            eprefill_host = eprefill_host,
            eprefill_port = eprefill_port,
            edecode_host = edecode_host,
            edecode_port = edecode_port,
            prefilled_token_id = prefilled_token_id,
            output_logprobs = output_logprobs,
            is_layer = is_layer
        )

        return stream
    
    async def add_prefilled_meta(self, request_id: str, prefilled_token_ids, output_logprobs):
        seq_group = self.engine.scheduler.kv_prepared_seq_group[request_id]
        for token_id, output_logprob in zip(prefilled_token_ids, output_logprobs):
            seq_group.get_seqs()[0].append_token_id(token_id, output_logprob)
        self.engine.scheduler.meta_recv_finished[request_id] = self.engine.scheduler.kv_prepared_seq_group[request_id]
        del self.engine.scheduler.kv_prepared_seq_group[request_id]
        
    async def prepare_layer_kv_blocks(self,
        layer_kv_blocks_meta,
    ) -> KvPreparedResponse:
        merge_request_id = random_uuid()
        merge_seq_groups = []
        merge_blocks = []
        merge_num_blocks = []
        merge_is_allocated = []
        global_ranks = None
        for meta in layer_kv_blocks_meta:
            request_id = meta["request_id"]
            prompt_token_ids = meta["prompt_token_ids"]
            global_ranks = meta["global_ranks"]
            eprefill_host = meta["eprefill_host"]
            eprefill_port = meta["eprefill_port"]
            edecode_host = meta["edecode_host"]
            edecode_port = meta["edecode_port"]
            
            sampling_params_json = meta["sampling_params"]
            sampling_params =  SamplingParams(**sampling_params_json)
            arrival_time = time.time()
            # Create the sequences.
            block_size = self.engine.cache_config.block_size
            seq_id = next(self.engine.seq_counter)
            seq = Sequence(seq_id, None, prompt_token_ids, block_size,
                        None, None)

            # Defensive copy of SamplingParams, which are used by the sampler,
            # this doesn't deep-copy LogitsProcessor objects
            sampling_params = sampling_params.clone()
            # inject the eos token id into the sampling_params to support min_tokens
            # processing
            sampling_params.eos_token_id = seq.eos_token_id
            # Create the sequence group.
            seq_group = SequenceGroup(request_id, [seq], sampling_params,
                                    arrival_time, None, None, None, eprefill_host=eprefill_host, eprefill_port=eprefill_port, edecode_host=edecode_host, edecode_port=edecode_port)
            can_allocate = self.engine.scheduler.block_manager.can_allocate(seq_group)
            if can_allocate == AllocStatus.OK:
                phy_blocks = self.engine.scheduler.allocate_kv_blocks(seq_group, True)
                blocks = [phy_block.block_number for phy_block in phy_blocks if phy_block.computed == False]
                computed_blocks = [phy_block.block_number for phy_block in phy_blocks if phy_block.computed == True]
                merge_blocks.append(blocks)
                merge_seq_groups.append(seq_group)
                merge_num_blocks.append(len(computed_blocks))
                merge_is_allocated.append(True)
                self.engine.scheduler.kv_prepared_seq_group[request_id] = seq_group
            else:
                merge_num_blocks.append(0)
                merge_is_allocated.append(False)
        if merge_seq_groups:
            self.engine.scheduler.recv_transfering[merge_request_id] = merge_seq_groups
            current_transfer_tag = self.engine.recv_kv_trans_scheduler.add_layer_kv_request(merge_request_id, global_ranks , merge_blocks)
        if not self.is_running:
            if self.start_engine_loop:
                self.start_background_loop()
                self._request_tracker.new_requests_event.set()
            else:
                raise AsyncEngineDeadError(
                    "Background loop is not running. If it was running, "
                    "inspect the output to find the stacktrace of the "
                    "error that caused the background loop to stop "
                    "(AsyncEngineDeadError).")
        else:
            self._request_tracker.new_requests_event.set()
        #TODO there may has some isuue when decode without hbm
        if merge_seq_groups:
            return merge_request_id, merge_num_blocks, current_transfer_tag, merge_is_allocated
        else:
            return None, 0, -1, False

    async def pull_kv_blocks(self, query_meta):
        self.engine.pull_kv_blocks(query_meta)
    
    async def query_kv_blocks(self, query_cache_meta):
        return self.engine.query_kv_blocks(query_cache_meta)

    async def add_kv_response(
        self,
        response: KvPreparedResponse,
    ) -> None:
        self._request_tracker.add_kv_response(response=response)
    
    async def generate(
        self,
        prompt: Optional[str]=None,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalData] = None,
        prefill_request_output: Optional[RequestOutput] = None,
        cache_meta: Optional[CacheMeta] = None,
        eprefill_host: Optional[str] = None,
        eprefill_port: Optional[str] = None,
        edecode_host: Optional[str] = None,
        edecode_port: Optional[str] = None,
        prefilled_token_id: Optional[List[int]] = None,
        output_logprobs: Optional[Dict[int, float]] = None,
        is_layer: Optional[bool] = False
    ) -> AsyncIterator[RequestOutput]:
        """Generate outputs for a request.

        Generate outputs for a request. This method is a coroutine. It adds the
        request into the waiting queue of the LLMEngine and streams the outputs
        from the LLMEngine to the caller.

        Args:
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters of the request.
            request_id: The unique id of the request.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            lora_request: LoRA request to use for generation, if any.
            multi_modal_data: Multi modal data per request.

        Yields:
            The output `RequestOutput` objects from the LLMEngine for the
            request.

        Details:
            - If the engine is not running, start the background loop,
              which iteratively invokes
              :meth:`~vllm.engine.async_llm_engine.AsyncLLMEngine.engine_step`
              to process the waiting requests.
            - Add the request to the engine's `RequestTracker`.
              On the next background loop, this request will be sent to
              the underlying engine.
              Also, a corresponding `AsyncStream` will be created.
            - Wait for the request outputs from `AsyncStream` and yield them.

        Example:
            >>> # Please refer to entrypoints/api_server.py for
            >>> # the complete example.
            >>>
            >>> # initialize the engine and the example input
            >>> engine = AsyncLLMEngine.from_engine_args(engine_args)
            >>> example_input = {
            >>>     "prompt": "What is LLM?",
            >>>     "stream": False, # assume the non-streaming case
            >>>     "temperature": 0.0,
            >>>     "request_id": 0,
            >>> }
            >>>
            >>> # start the generation
            >>> results_generator = engine.generate(
            >>>    example_input["prompt"],
            >>>    SamplingParams(temperature=example_input["temperature"]),
            >>>    example_input["request_id"])
            >>>
            >>> # get the results
            >>> final_output = None
            >>> async for request_output in results_generator:
            >>>     if await request.is_disconnected():
            >>>         # Abort the request if the client disconnects.
            >>>         await engine.abort(request_id)
            >>>         # Return or raise an error
            >>>         ...
            >>>     final_output = request_output
            >>>
            >>> # Process and return the final output
            >>> ...
        """
        # Preprocess the request.
        arrival_time = time.time()

        try:
            stream = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                prompt_token_ids=prompt_token_ids,
                arrival_time=arrival_time,
                lora_request=lora_request,
                multi_modal_data=multi_modal_data,
                prefill_request_output=prefill_request_output,
                cache_meta = cache_meta,
                eprefill_host = eprefill_host,
                eprefill_port = eprefill_port,
                edecode_host=edecode_host,
                edecode_port = edecode_port,
                prefilled_token_id=prefilled_token_id,
                output_logprobs = output_logprobs,
                is_layer = is_layer
            )

            async for request_output in stream:
                yield request_output
        except (Exception, asyncio.CancelledError) as e:
            # If there is an exception or coroutine is cancelled, abort the
            # request.
            self._abort(request_id)
            raise e

    async def abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        if not self.is_running:
            raise AsyncEngineDeadError(
                "Background loop is not running. If it was running, "
                "inspect the output to find the stacktrace of the "
                "error that caused the background loop to stop "
                "(AsyncEngineDeadError).")

        return self._abort(request_id)

    def _abort(self, request_id: str) -> None:
        """Abort a request.

        Abort a submitted request. If the request is finished or not found,
        this method will be a no-op.

        Args:
            request_id: The unique id of the request.
        """
        self._request_tracker.abort_request(request_id,
                                            verbose=self.log_requests)

    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        if self.engine_use_ray:
            return await self.engine.get_model_config.remote()
        else:
            return self.engine.get_model_config()

    async def do_log_stats(self) -> None:
        if self.engine_use_ray:
            await self.engine.do_log_stats.remote()
        else:
            self.engine.do_log_stats()

    async def check_health(self) -> None:
        """Raises an error if engine is unhealthy."""
        t = time.perf_counter()
        logger.debug("Starting health check...")
        if self.is_stopped:
            raise AsyncEngineDeadError("Background loop is stopped.")

        if self.engine_use_ray:
            try:
                await self.engine.check_health.remote()
            except ray.exceptions.RayActorError as e:
                raise RuntimeError("Engine is dead.") from e
        else:
            await self.engine.check_health_async()
        logger.debug(f"Health check took {time.perf_counter()-t}s")

    async def get_nccl_id(self, dst_channel, worker_type) -> None:
        nccl_id = await self.engine.model_executor._run_driver_async("get_nccl_id", dst_channel=dst_channel, worker_type=worker_type)
        res = await self.engine.model_executor._run_workers_async("create_comm", nccl_id=nccl_id, dst_channel=dst_channel,worker_type=worker_type)
        return nccl_id
    
    async def create_comm(self, nccl_id, dst_channel, worker_type) -> None:
        if worker_type == "sender":
            res = await self.engine.model_executor._run_workers_async("create_comm", nccl_id=nccl_id, dst_channel=dst_channel, worker_type="receiver")
        else:
            res = await self.engine.model_executor._run_workers_async("create_comm", nccl_id=nccl_id, dst_channel=dst_channel, worker_type="sender")
    