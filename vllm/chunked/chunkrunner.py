from transformers import PreTrainedTokenizerBase, AutoModelForSequenceClassification, AutoTokenizer
import json
from typing import List, Dict, Tuple, Any
import torch
import time
import random

from vllm.config import ModelConfig, ParallelConfig
from vllm.chunked.chunkworker import ChunkWorker
from vllm.chunked.chunkcache import ChunkCacheBlocks
from vllm.chunked.chunk import Chunk, ChunkInputMetadata, ChunkSamplingParams, ChunkStatus, Sequence
from vllm.worker.worker import _pad_to_max
from vllm.engine.ray_utils import initialize_cluster, ray
from vllm.utils import random_uuid, Counter
import threading

class ChunkRunner:
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 chunk_size: int,
                 chunk_num: int) -> None:
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_num = chunk_num
        self.all_total_sequences: List[Sequence] = []
        self.all_job_sequences: Dict[str, Sequence] = {}
        self.all_job_chunks: List[Chunk] = []
        self.processed_chunks: List[Chunk] = []
        self.counter = Counter()
        self.cacheblock = ChunkCacheBlocks(blocks_num = self.chunk_num)

        self.request_waiting  = [[],[], [], []]
        
    def set_predict_model_and_tokenizer(self, 
                                        predict_tokenizer_path: str, 
                                        predict_model_path: str) -> None:
        self.predict_tokenizer = AutoTokenizer.from_pretrained(predict_tokenizer_path)
        self.predict_model = AutoModelForSequenceClassification.from_pretrained(predict_model_path,
                                                                                num_labels = 10)
        # self.predict_model = self.predict_model.to("cuda")
        
    def monitor_mprefill_info(self):
        unfinished_chunked_token = 0
        for job_chunk in self.all_job_chunks:
            if job_chunk.chunk_status == ChunkStatus.RUNNING or job_chunk.chunk_status == ChunkStatus.WAITING:
                unfinished_chunked_token = unfinished_chunked_token + len(job_chunk.prompt_token_ids)
        return unfinished_chunked_token
    
    def add_requests_to_job_sequences(self,
                                      prompts_s,
                                      prompt_token_ids_s: List[List[int]],
                                      sampling_params_s: List[ChunkSamplingParams],
                                      request_ids: List[str],
                                      request_label: Dict[str, int]) -> None:
        # seq_ids: List[str] = []
        # for _ in range(len(sampling_params_s)):
        #     seq_ids.append(random_uuid())

        start  = time.time()
        # this_labels = self._do_predict(inputs = prompts_s,
        #                                request_ids = request_ids, request_label = request_label)
        
        #mock predict model result
        for request_id  in request_ids:
            request_label[request_id] = 0
        this_labels = [0] * len(request_ids)
        
        end = time.time()
        # print("this labels ", this_labels , end-start)
        for prompt_token_ids, sampling_params, request_id, label in zip(prompt_token_ids_s, sampling_params_s, request_ids, this_labels):
            # seq_id = random_uuid()
            now_time = time.time()
            a_sequence = Sequence(seq_id = request_id, 
                                  prompt_token_ids = prompt_token_ids,
                                  sampling_params = sampling_params,
                                  start_time = now_time,
                                  request_id=request_id,
                                   label = label)
            self.all_job_sequences[request_id] = a_sequence
        self._set_job_chunks()

    def _add_requests(self, 
                      prompt_token_ids: List[int], 
                      sampling_params: ChunkSamplingParams) -> None:
        seq_id = random_uuid()
        now_time = time.time()
        self.all_total_sequences.append(Sequence(seq_id = seq_id, 
                                             prompt_token_ids = prompt_token_ids,
                                             sampling_params = sampling_params,
                                             start_time = now_time)) 

    def set_self_configs(self, model: str, tensor_parallel_size: int) -> None:
        model_config = ModelConfig(model = model, tokenizer = None, tokenizer_mode = 'auto', 
                                   trust_remote_code = False, download_dir = None, use_np_weights = False,
                                   use_dummy_weights = False, dtype = 'auto', seed = 0)
        self.model_config = model_config
        if tensor_parallel_size > 1:
            worker_use_ray = True
        else:
            worker_use_ray = False
        self.worker_use_ray = worker_use_ray
        self.parallel_config = ParallelConfig(pipeline_parallel_size = 1,
                                              tensor_parallel_size = tensor_parallel_size,
                                              worker_use_ray = self.worker_use_ray)
        self._set_self_ray_environment()
    
    def _set_self_ray_environment(self) -> None:
        distributed_init_method, devices = initialize_cluster(parallel_config = self.parallel_config,
                                                              engine_use_ray = self.worker_use_ray)
        self.distributed_init_method = distributed_init_method
        self.devices = devices
    
    def set_self_chunkworker(self, chunk_size: int, chunk_num: int) -> None:
        chunk_worker = ChunkWorker(chunk_size = chunk_size, chunk_num = chunk_num, 
                                   model_config = self.model_config,
                                   parallel_config = self.parallel_config,
                                   rank = 0,
                                   distributed_init_method = self.distributed_init_method)
        self.chunk_worker = chunk_worker
    
    def set_parallel_chunkworkers(self) -> None:
        self.workers: List[ChunkWorker] = []
        assert len(self.devices) == 1, "PP is under coding"
        for rank, node_resource, _ in self.devices[0]:
            worker_cls = ChunkWorker
            if self.worker_use_ray:
                worker_cls = ray.remote(num_cpus = 0,
                                        num_gpus = 1,
                                        resources = {node_resource: 1e-3})(worker_cls).remote
            worker = worker_cls(chunk_size = self.chunk_size,
                                chunk_num = self.chunk_num,
                                model_config = self.model_config,
                                parallel_config = self.parallel_config,
                                rank = rank,
                                distributed_init_method = self.distributed_init_method)
            self.workers.append(worker)
                
    
    def set_inputs(self, dataset_path: str, num_requests: int) -> None:
        with open(dataset_path) as f:
            dataset = json.load(f)
        dataset = [
            data for data in dataset
            if len(data["conversations"]) >= 2
        ]
        dataset = [
            (data["conversations"][0]["value"], data["conversations"][1]["value"])
            for data in dataset
        ]
        prompts = [prompt for prompt, _ in dataset]
        prompt_token_ids = self.tokenizer(prompts).input_ids
        completions = [completion for _, completion in dataset]
        completion_token_ids = self.tokenizer(completions).input_ids
        tokenized_dataset = []
        for i in range(len(dataset)):
            output_len = len(completion_token_ids[i])
            tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))
        filtered_dataset = []
        num_requests_count = 0
        for _, prompt_token_ids, output_len in tokenized_dataset:
            prompt_len = len(prompt_token_ids)
            if prompt_len < 4 or output_len < 4:
                continue
            filtered_dataset.append(prompt_token_ids)
            num_requests_count += 1
            if num_requests_count == num_requests:
                break
        self.requests = filtered_dataset
    
    def _test_time_for_mixed(self, data: List[int]) -> float:
        st = time.time()
        #data.sort()
        ans = 0
        ans2 = []
        slit = []
        temp = []
        data2 = []
        data3 = []
        for i,x in enumerate(data):
            if x >= 512:
                data2.append(x)
            else:
                data3.append(x)
        for i,x in enumerate(data3):
            ans += x
            temp.append(x)
            if ans >= 256 or i == (len(data3)-1):
                ans = 0
                slit.append(temp)
                ans2.append(sum(temp))
                temp = []
        #print(slit)
        last = {}
        for x2 in data2:
            t = x2
            for i, sums in enumerate(ans2):
                if sums >= 512:
                    continue
                n = 512 - sums
                if x2 >= n:
                    slit[i].append(n)
                    x2 -= n
                    ans2[i] += n
                else:
                    slit[i].append(x2)
                    ans2[i] += x2
                    x2 = 0
                    print(f"{t} in {i}")
                    break
            if x2 != 0:
                last[t] = [x2]
        for l,d in last.items():
            slit.append(d)
            print(f"{l} in {len(slit)-1}")
        data_f = []
        for s in slit:
            data_f.extend(s)
        ed = time.time()
        return ed - st

    def _add_requests_to_self(self) -> None:
        #for prompt_token_ids in self.requests:
        random.seed(42)
        for _ in range(3):
            cold_start_token_ids = [random.randint(1, 9) for _ in range(self.chunk_size)]
            cold_start_sampling_params = ChunkSamplingParams(temperature = 0, top_p = 1.0, top_k = -1)
            self._add_requests(prompt_token_ids = cold_start_token_ids, 
                                        sampling_params = cold_start_sampling_params)
        #prompt_lens = [91,88,75,3,255,42,103,15,352,385,127,49,306,157,228,75,209,315,197,21,104,30,182,175,103,53,
        #               356,390,122,387,125,22,347,143,273,239,117,276,119,422,90,269,243,243,24,29,216,88,93,107,224,
        #               248,75,189,108,72,108,88,264,248,339,173,328,184,62,47,2,76,61,178,86,52,60,2,89,59,21,283,92,
        #               125,295,497,15,64,235,213,505,7,189,3,320,345,167,58,38,87,15,314,344,168,100,83,22,191,116,130,
        #               62,80,44,196,33,122,308,49,626,546,2047,1845,878,939,775,1021,592,513,1020,955,931,1795,1500,942,
        #               899,1413,561]
        '''prompt_lens = [2, 2, 3, 15, 15, 21, 21, 22, 22, 24, 30, 33, 38,264, 42, 47, 49, 52, 53, 58, 211,
        59, 60, 60, 61, 62, 62, 210,64, 72, 75, 75,239,75,76,80,83,42,156,87,88,88,249,89,91,92,240,93,100,103,216,
        103,104,107,198,108,108,117,179,122,125,130,135,178,182,152,189,191,132,228,235,49,243,248,21,264,248,269,72,
        273,239,276,236,306,71,308,204,315,197,328,184,339,173,344,168,345,167,347,165,385,127,387,69,56,390,122,422,90,
        497,15,505,7,839,871, 546, 2047, 1845, 878, 939, 775, 1021, 592, 513, 1020, 955, 931, 1795, 1500, 942, 899, 
        1413, 561]'''
        #sts = time.time()
        #prompt_lens.sort()
        #eds = time.time()
        #fake_lens = [91,88,75,3,255,42,103,15,352,385,127,49,306,157,228,75,209,315,197,21,104,30,182,175,103,53,
        #               356,390,122,387,125,22,347,143,273,239,117,276,119,422,90,269,243,243,24,29,216,88,93,107,224,
        #               248,75,189,108,72,108,88,264,248,339,173,328,184,62,47,2,76,61,178,86,52,60,2,89,59,21,283,92,
        #               125,295,497,15,64,235,213,505,7,189,3,320,345,167,58,38,87,15,314,344,168,100,83,22,191,116,130,
        #               62,80,44,196,33,122,308,49,626,546,2047,1845,878,939,775,1021,592,513,1020,955,931,1795,1500,942,
        #               899,1413,561]
        #random.seed(2023)
        #random.shuffle(prompt_lens)
        #print(prompt_lens)
        #time_slot = self._test_time_for_mixed(data = fake_lens)
        #print(f"sort costs {eds - sts} seconds")
        #print(f"mixed costs {time_slot} seconds")
        #for prompt_len in prompt_lens:
        #    sampling_params = ChunkSamplingParams(temperature = 0, top_p = 1.0, top_k = -1)
        #    #self.chunk_worker.add_requests(prompt_token_ids = prompt_token_ids, sampling_params = sampling_params)
        #    dummy_prompt_token_ids = [random.randint(1, 9) for _ in range(prompt_len)]
        #    self._add_requests(prompt_token_ids = dummy_prompt_token_ids, sampling_params = sampling_params)
    
    def _start_worker(self) -> None:
        self._add_requests_to_self()
        self._set_job_sequences()
        self._set_job_chunks()
        #self.chunk_worker.set_job_sequences()
        #self.chunk_worker.set_job_chunks()

    def _prepare_model_inputs(self, 
                              chunk: Chunk) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, List[Tuple[int, int, int]]]]:
        chunk_size = self.chunk_size
        input_tokens = chunk.prompt_token_ids
        input_positions: List[int] = []
        for i, slice_length in enumerate(chunk.prompt_lens):
            slice_positions: List[int] = list(range(slice_length))
            slice_positions = [x + chunk.kv_prefixs[i] for x in slice_positions]
            input_positions.extend(slice_positions)
        input_tokens = _pad_to_max(input_tokens, max_len = chunk_size)
        input_tokens_tensor = torch.cuda.LongTensor(input_tokens)
        input_positions = _pad_to_max(input_positions, max_len = chunk_size)
        input_positions_tensor = torch.cuda.LongTensor(input_positions)
        kv_cache_ids: Dict[int, List[Tuple[int, int, int]]] = {} # prompt_len to (block, st, length)s
        for seq_id, kv_cache_num in chunk.seqs_to_prefixs.items():
            sequence = self.all_job_sequences[seq_id]
            if kv_cache_num == 0:
                prompt_len = chunk.seqs_to_lens.get(seq_id)
                kv_cache_ids.setdefault(prompt_len, [])
            else:
                for chunk_id, used_token_num in sequence.chunks_to_prompts.items():
                    if chunk_id == chunk.chunk_id:
                        break
                    else:
                        if chunk_id >= 0 and chunk_id < len(self.all_job_chunks):
                            block_id = self.all_job_chunks[chunk_id].cache_block_id
                            st = 0
                            for a_seq_id, slice_token_num in self.all_job_chunks[chunk_id].seqs_to_lens.items():
                                if a_seq_id == seq_id:
                                    break
                                else:
                                    st += slice_token_num
                            prompt_len = chunk.seqs_to_lens.get(seq_id)
                            kv_cache_ids.setdefault(prompt_len, []).append((block_id, st, used_token_num))
                        else:
                            print(f"chunk id is {chunk_id}")
                            print(f"has {len(self.all_job_chunks)} chunks")
        return (input_tokens_tensor, input_positions_tensor, kv_cache_ids)
    
    def run_worker(self) -> None:
        self._start_worker()

        #now_time = time.time()
        #print(f"Added in working pool at {now_time}")
        
        for chunk in  self.all_job_chunks: #for chunk in self.chunk_worker.job_chunks:
            #chunk = self.all_job_chunks[0]
            start_time = time.time()

            chunk.chunk_status = ChunkStatus.RUNNING

            input_tokens_tensor, input_positions_tensor, kv_cache_ids = self._prepare_model_inputs(chunk)

            chunkinputmetadata = ChunkInputMetadata(prompt_lens = chunk.prompt_lens, 
                                                    kv_prefixs = chunk.kv_prefixs,
                                                    kv_prefixs_blocks = kv_cache_ids, 
                                                    kv_block = chunk.cache_block_id,
                                                    idxs = chunk.idxs,
                                                    sampling_params_for_sampler = chunk.sampling_params_for_sampler,
                                                    do_cat = chunk.do_cat)
            
            output_token_list, logprobs = self._run_workers("execute_model",
                                        inputs = input_tokens_tensor,
                                        inputs_positions = input_positions_tensor,
                                        #kv_cache = self.chunk_worker.kv_cache,
                                        chunkmetadata = chunkinputmetadata)

            '''output = output[idxs]
            output_token_list, logprobs = self._run_workers("execute_sampler",
                                                            logits = output, 
                                                            sampling_params = sampling_params)'''
            
            end_time = time.time()
            for i, id in enumerate(chunk.do_sampling):
                self.all_job_sequences[id].add_first_token_id(output_token_list[i])
                self.all_job_sequences[id].add_first_token_logprob(logprobs[i])
                self.all_job_sequences[id].set_end_time(st = start_time,
                                                        ed = end_time)
                #self.chunk_worker.job_sequences[id].add_first_token_id(output_token_list[i])
                #self.chunk_worker.job_sequences[id].add_first_token_logprob(logprobs[i])
                #self.chunk_worker.job_sequences[id].set_end_time(end_time)

            #self.processed_chunks.append(chunk)
            #self.all_job_chunks.pop(0)

        self._reduce_outputs()
        #self.all_job_chunks.clear()
        #self.chunk_worker.reduce_outputs()
        #self.chunk_worker.generate_first_token_id()
        #self.chunk_worker.generate_first_token_str(tokenizer = self.tokenizer)
    
    def _run_workers(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        for worker in self.workers:
            executor = getattr(worker, method)
            if self.parallel_config.worker_use_ray:
                executor = executor.remote

            output = executor(*args, **kwargs)
            all_outputs.append(output)

        if self.parallel_config.worker_use_ray:
            all_outputs = ray.get(all_outputs)
       
        if get_all_outputs:
            return all_outputs

        # Make sure all workers have the same results.
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        
        return output
    
    def _set_job_sequences(self) -> None:
        total_token_num = self.chunk_num * self.chunk_size
        count = 0
        while len(self.all_total_sequences) > 0:
            sequence = self.all_total_sequences[0]
            if count + sequence.prompt_len <= total_token_num:
                sequence = self.all_total_sequences.pop(0)
                count += sequence.prompt_len
                self.all_job_sequences[sequence.seq_id] = sequence
            else:
                break
    
    def _set_job_chunks(self) -> None:
        self.counter.reset()
        all_token_ids: List[int] = []
        all_token_seqs: List[str] = []
        for _, sequence in self.all_job_sequences.items():
            if sequence.processed:
                continue
            else:
                sequence.processed = True
                for token_id in sequence.prompt_token_ids:
                    all_token_ids.append(token_id)
                    all_token_seqs.append(sequence.seq_id)
        st = 0
        token_num = len(all_token_ids)
        while st < token_num:
            ed = st + self.chunk_size
            if ed >= token_num:
                ed = token_num
            chunk_id = self.counter.__next__()
            chunk = Chunk(chunk_id = chunk_id, chunk_size = self.chunk_size, 
                          chunk_status = ChunkStatus.WAITING)
            temp_seqtoken_count: Dict[str, int] = {}
            for i in range(st, ed):
                chunk.prompt_token_ids.append(all_token_ids[i])
                temp_seqtoken_count[all_token_seqs[i]] = temp_seqtoken_count.setdefault(all_token_seqs[i], 0) + 1
                self.all_job_sequences[all_token_seqs[i]].chunks_to_prompts[chunk_id] = self.all_job_sequences[all_token_seqs[i]].chunks_to_prompts.setdefault(chunk_id, 0) + 1
            for temp_seq_id, temp_token_len in temp_seqtoken_count.items():
                chunk.raw_sequence_ids.append(temp_seq_id)
                chunk.prompt_lens.append(temp_token_len)
                ans = 0
                for temp_chunk_id, used_token_num in self.all_job_sequences[temp_seq_id].chunks_to_prompts.items():
                    if temp_chunk_id == chunk_id:
                        break
                    else:
                        ans += used_token_num
                chunk.kv_prefixs.append(ans)
            chunk.set_seqs_to_lens_and_prefixs()
            temp_block = self.cacheblock.allocate_block()
            chunk.set_self_block(block = temp_block)
            self.all_job_chunks.append(chunk)
            st += self.chunk_size
        
        for chunk in self.all_job_chunks:
            st = 0
            idxs: List[int] = []
            sampling_params: List[ChunkSamplingParams] = []
            do_sampling: List[str] = []
            for seq_id, prompt_len in chunk.seqs_to_lens.items():
                ed = st + prompt_len
                self.all_job_sequences[seq_id].update_count(prompt_len)
                #self.chunk_worker.job_sequences[seq_id].update_count(prompt_len)
                if self.all_job_sequences[seq_id].is_full(): #if self.chunk_worker.job_sequences[seq_id].is_full():
                    idxs.append(ed - 1)
                    do_sampling.append(seq_id)
                    sampling_params.append(self.all_job_sequences[seq_id].sampling_params)
                    #sampling_params.append(self.chunk_worker.job_sequences[seq_id].sampling_params)
                #self.chunk_worker.job_sequences[seq_id].outputs.append(output[st: ed])
                st = ed
                #self.chunk_worker.job_sequences[seq_id].add_start_and_end_time(st = start_time, ed = end_time)
                #st = ed
            chunk.set_idxs(idxs = idxs)
            chunk.set_sampling_params_for_sampler(sampling_params_for_sampler = sampling_params)
            chunk.set_do_sampling(do_sampling = do_sampling)
    
    def _reduce_outputs(self) -> None:
        #for _, sequence in self.job_sequences.items():
        #    for i in range(len(sequence.outputs)):
        #        if i != 0:
        #            sequence.outputs[0] = torch.cat((sequence.outputs[0], sequence.outputs[i]), 0)
        # free all chunks' cache
        for chunk in self.all_job_chunks:
            #chunk = self.processed_chunks.pop(0)
            self.cacheblock.free_block(block = chunk.cache_block)
            chunk.chunk_status = ChunkStatus.PREFILLED
        self.all_job_chunks.clear()

    #todo
    def find_decode_host(self,mdecode_info):
        return 
    
    def write_to_mdispatcher(self, prefill_nums, num, request_id, label, mm):
        combined_info_bytes = num.to_bytes(1, byteorder='big') + request_id.encode("utf-8") + label.to_bytes(1, byteorder='big') + prefill_nums.to_bytes(1, byteorder='big')
        # print("combined_info_bytes ", len(combined_info_bytes), combined_info_bytes, request_id, time.time())
        start = time.time()
        mm.seek((prefill_nums-1)*35)
        mm.write(combined_info_bytes)
        end = time.time()
        print("write_to_mdispatcher ", request_id, start, end , start-end)
        
    def mprefill_generate_prefill(self, mm, prefill_nums, request_label, mdecode_info) -> int:
        #self._set_job_chunks()
        output_num = 0
        sended_request_id = set()
        # pass_time = 0
        num = 1
        for chunk in self.all_job_chunks:
            start_time = time.time()
            chunk.chunk_status = ChunkStatus.RUNNING
            input_tokens_tensor, input_positions_tensor, kv_cache_ids = self._prepare_model_inputs(chunk)
            chunkinputmetadata = ChunkInputMetadata(prompt_lens = chunk.prompt_lens, 
                                                    kv_prefixs = chunk.kv_prefixs,
                                                    kv_prefixs_blocks = kv_cache_ids, 
                                                    kv_block = chunk.cache_block_id,
                                                    idxs = chunk.idxs,
                                                    sampling_params_for_sampler = chunk.sampling_params_for_sampler,
                                                    do_cat = chunk.do_cat)
            output_token_list, logprobs = self._run_workers("execute_model",
                                        inputs = input_tokens_tensor,
                                        inputs_positions = input_positions_tensor,
                                        chunkmetadata = chunkinputmetadata)
            end_time = time.time()
            output_num += len(chunk.do_sampling)
            for i, id in enumerate(chunk.do_sampling):
                self.all_job_sequences[id].add_first_token_id(output_token_list[i])
                self.all_job_sequences[id].add_first_token_logprob(logprobs[i])
                self.all_job_sequences[id].set_end_time(st = start_time, ed = end_time)
            

            
            for request_id in chunk.do_sampling:
                if request_id not in sended_request_id:
                    label = request_label.get(request_id)
                    # print(request_id, label)
                    self.find_decode_host(mdecode_info)
                    prefill_nums += 1
                    threading.Thread(target=self.write_to_mdispatcher, args=(prefill_nums, num, request_id
                                                                             , label ,mm)).start()
                    #put in thread
                    # combined_info_bytes = prefill_nums.to_bytes(1, byteorder='big') + num.to_bytes(1, byteorder='big') + request_id.encode("utf-8") + label.to_bytes(1, byteorder='big')
                    # print("combined_info_bytes ", len(combined_info_bytes))
                    # mm.seek((prefill_nums-1)*35)
                    # mm.write(combined_info_bytes)
                    sended_request_id.add(request_id)
                    # time.sleep(0.000005)
            #self.processed_chunks.append(chunk)
        #self.all_job_chunks.clear()
        self._reduce_outputs()

        print("mprefill!!:  prefill iteration now is no unfinished")
        return prefill_nums       
    
        
    
    def _do_predict(self, inputs: List[str], request_ids: List[str], request_label: Dict[str, int]) -> List[int]:
        test_encoded = self.predict_tokenizer(inputs, 
                                                padding = "max_length", 
                                                truncation = True, 
                                                return_tensors = "pt", 
                                                max_length = 2048)
        # test_encoded = {key: value.to("cuda") for key, value in test_encoded.items()}
        predictions = self.predict_model(input_ids = test_encoded['input_ids'], 
                                            attention_mask = test_encoded['attention_mask'])
        predicted_labels = torch.argmax(predictions.logits, dim = 1).tolist()
        for request_id, label in zip(request_ids, predicted_labels):
            # print(f"{request_id}'s label is {label}", type(label))
            request_label[request_id] = label
        print("request_label " , request_label)
        return predicted_labels