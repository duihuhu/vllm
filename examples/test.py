import torch
import threading
import json
#from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple
import time

from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.chunked.chunkrunner import ChunkRunner
from vllm.chunked.chunk import ChunkInputMetadata, ChunkSamplingParams
from vllm.worker.worker import _pad_to_max #,_pad_to_alignment

'''def _pad_to_max_for_predict(x: List[int], max_len: int) -> List[int]:
    return x + [1] * (max_len - len(x))'''

def set_inputs(tokenizer, dataset_path: str, num_requests: int):
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
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))
    filtered_dataset = []
    num_requests_count = 0
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            continue
        if prompt_len > 2048 or output_len > 2048:
            continue
        filtered_dataset.append((prompt, prompt_token_ids, output_len))
        num_requests_count += 1
        if num_requests_count == num_requests:
            break
    return filtered_dataset

def execute_big_model_warm(input_tokens_ids_tensors: List[torch.Tensor], 
                      input_positions_tensors: List[torch.Tensor], 
                      input_chunkinputmetadata: List[ChunkInputMetadata]) -> None:
    iter = 0
    iter_time = []
    for input_tokens_ids_tensor, input_positions_tensor, chunkinputmetadata in zip(input_tokens_ids_tensors, 
                                                                                   input_positions_tensors, 
                                                                                   input_chunkinputmetadata):
        st = time.time()
        _, _ = chunkrunner_13b._run_workers("execute_model",
                                                inputs = input_tokens_ids_tensor,
                                                inputs_positions = input_positions_tensor,
                                                chunkmetadata = chunkinputmetadata)
        ed = time.time()
        iter_time.append(ed-st)
        # with open("/workspace/vllm/examples/logs/co_running_l_512_1.txt", 'a') as file:
        #     file.write(f"iter {iter}, start at {st}, end at {ed}, costs {ed - st} seconds\n")
        iter += 1
        #print(f"output_token_list: {output_token_list}")
        #print(f"logprobs: {logprobs}")

def execute_big_model(input_tokens_ids_tensors: List[torch.Tensor], 
                      input_positions_tensors: List[torch.Tensor], 
                      input_chunkinputmetadata: List[ChunkInputMetadata]) -> None:
    iter = 0
    iter_time = []
    start_time = time.time()
    for input_tokens_ids_tensor, input_positions_tensor, chunkinputmetadata in zip(input_tokens_ids_tensors, 
                                                                                   input_positions_tensors, 
                                                                                   input_chunkinputmetadata):
        st = time.time()
        _, _ = chunkrunner_13b._run_workers("execute_model",
                                                inputs = input_tokens_ids_tensor,
                                                inputs_positions = input_positions_tensor,
                                                chunkmetadata = chunkinputmetadata)
        ed = time.time()
        iter_time.append(ed-st)
        # with open("/workspace/vllm/examples/logs/co_running_l_512_1.txt", 'a') as file:
        #     file.write(f"iter {iter}, start at {st}, end at {ed}, costs {ed - st} seconds\n")
        iter += 1
        #print(f"output_token_list: {output_token_list}")
        #print(f"logprobs: {logprobs}")
    end_time = time.time()
    print("big model total time " , start_time, end_time, end_time-start_time)
    for i_time in iter_time:
        print(i_time)

def execute_small_model_warm(input_prompts: List[Tuple[str, int]]
                        #input_positions_tensor, 
                        #chunkinputmetadata,
                        ) -> None:
    iter = 0
    iter_time = []
    # for i in range(6):
    for input_prompt, input_prompt_len in input_prompts:
        st = time.time()
        _ = chunkrunner_125m.execute_predict_model(input_prompt, 512)
        '''predict_labels = chunkrunner_125m._run_workers("execute_predict_model",
                                    inputs = input_tokens_ids_tensor,
                                    inputs_positions = input_positions_tensor,
                                    chunkmetadata = chunkinputmetadata)'''
        ed = time.time()
        iter_time.append(ed-st)
        # with open("/workspace/vllm/examples/logs/co_running_s_512_1.txt", 'a') as file:
        #     file.write(f"iter {iter}, start at {st}, end at {ed}, costs {ed - st} seconds\n")
        iter += 1

    
def execute_small_model(input_prompts: List[Tuple[str, int]]
                        #input_positions_tensor, 
                        #chunkinputmetadata,
                        ) -> None:
    iter = 0
    iter_time = []
    start_time = time.time()
    # for i in range(6):
    for input_prompt, input_prompt_len in input_prompts:
        st = time.time()
        _ = chunkrunner_125m.execute_predict_model(input_prompt, 512)
        '''predict_labels = chunkrunner_125m._run_workers("execute_predict_model",
                                    inputs = input_tokens_ids_tensor,
                                    inputs_positions = input_positions_tensor,
                                    chunkmetadata = chunkinputmetadata)'''
        ed = time.time()
        iter_time.append(ed-st)
        # with open("/workspace/vllm/examples/logs/co_running_s_512_1.txt", 'a') as file:
        #     file.write(f"iter {iter}, start at {st}, end at {ed}, costs {ed - st} seconds\n")
        iter += 1
        #print(f"predict costs {ed - st} seconds")
        #print(f"predict_labels: {predict_labels}")  
    end_time = time.time()
    print("small model total time " , start_time, end_time, end_time-start_time)
    for i_time in iter_time:
        print(i_time)
if __name__ == "__main__":

    tokenizer_13b = get_tokenizer("/workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/")
    #tokenizer_13b = get_tokenizer("/workspace/models/facebook/opt-125m")
    chunkrunner_13b = ChunkRunner(tokenizer = tokenizer_13b,
                              chunk_size = 512,
                              chunk_num = 10)
    chunkrunner_13b.set_self_configs(#model = "/workspace/models/facebook/opt-125m",
                            model = "/workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/",
                                 predict_model = None,
                                 tensor_parallel_size = 2)
    chunkrunner_13b.set_parallel_chunkworkers(num_gpus = 0.7)

    #tokenizer_125m = get_tokenizer("/workspace/opt-125m")
    #tokenizer_125m = get_tokenizer("/workspace/models/facebook/opt-125m")
    chunkrunner_125m = ChunkRunner(#tokenizer = tokenizer_125m,
                              tokenizer = None,
                              chunk_size = 512,
                              chunk_num = 10)
    chunkrunner_125m.set_predict_model_and_tokenizer(predict_tokenizer_path = "/workspace/opt-125m",
                                                     predict_model_path = "/workspace/opt_125m_model_sharegpt")
    '''chunkrunner_125m.set_self_configs(model = None,
                                 predict_model = "/workspace/opt_125m_model_sharegpt",
                                 tensor_parallel_size = 2)
    chunkrunner_125m.set_parallel_chunkworkers(num_gpus = 0.3)'''


    '''predict_tokenizer = AutoTokenizer.from_pretrained("/workspace/opt-125m")
    #predict_tokenizer = AutoTokenizer.from_pretrained("/workspace/models/facebook/opt-125m")
    predict_model = AutoModelForSequenceClassification.from_pretrained("/workspace/opt_125m_model_sharegpt", num_labels = 10)
    predict_model = predict_model.to('cuda:2')'''
    
    filtered_dataset = set_inputs(tokenizer = tokenizer_13b,
                                  dataset_path = "/workspace/ShareGPT_V3_unfiltered_cleaned_split.json",
                                  #dataset_path = "/workspace/datasets/ShareGPT_V3_unfiltered_cleaned_split.json",
                                  num_requests = 128)
    
    input_prompts: List[Tuple[str, int]] = []
    input_tokens_ids_tensors: List[torch.Tensor] = []
    input_positions_tensors: List[torch.Tensor] = []
    input_chunkinputmetadata: List[ChunkInputMetadata] = []

    for input_prompt, input_tokens_ids, output_len in filtered_dataset:
        input_prompts.append((input_prompt, len(input_tokens_ids)))
        #print(f"output_len is {output_len}")

        #small_input_tokens_ids = input_tokens_ids
        #small_input_tokens_ids = _pad_to_max_for_predict(small_input_tokens_ids, max_len = 2048)
        if len(input_tokens_ids) < 512:
            input_tokens_ids = _pad_to_max(input_tokens_ids, max_len = 512)
        else:
            input_tokens_ids = input_tokens_ids[0: 512]
        #input_positions = list(range(len(input_tokens_ids)))
        input_positions = list(range(512))
        input_tokens_ids_tensor = torch.cuda.LongTensor(input_tokens_ids)
        input_positions_tensor = torch.cuda.LongTensor(input_positions)
        chunkinputmetadata = ChunkInputMetadata(#prompt_lens = [len(input_tokens_ids)],
                                            prompt_lens = [512],
                                            kv_prefixs = [0],
                                            kv_prefixs_blocks = None,
                                            kv_block = None,
                                            #idxs = [len(input_tokens_ids) - 1],
                                            idxs = [511],
                                            sampling_params_for_sampler = [ChunkSamplingParams(temperature = 0.0,
                                                                                            top_p = 1.0,
                                                                                            top_k = -1)],
                                            do_cat = False)
        
        input_tokens_ids_tensors.append(input_tokens_ids_tensor)
        input_positions_tensors.append(input_positions_tensor)
        input_chunkinputmetadata.append(chunkinputmetadata)
    
    thread_big = threading.Thread(target = execute_big_model_warm, 
                                    args = (input_tokens_ids_tensors, 
                                            input_positions_tensors, 
                                            input_chunkinputmetadata))

    thread_big.start()
    thread_small = threading.Thread(target = execute_small_model_warm, args = (input_prompts,))
    thread_small.start()
    thread_small.join()
    thread_big.join()

    
    
    thread_big = threading.Thread(target = execute_big_model, 
                                    args = (input_tokens_ids_tensors, 
                                            input_positions_tensors, 
                                            input_chunkinputmetadata))
    thread_big.start()
    thread_small = threading.Thread(target = execute_small_model, args = (input_prompts,))
    thread_small.start()

    thread_big.join()
    thread_small.join()

    '''small_input_positions = list(range(len(small_input_tokens_ids)))
        small_input_tokens_ids_tensor = torch.cuda.LongTensor(small_input_tokens_ids)
        small_input_positions_tensor = torch.cuda.LongTensor(small_input_positions)
        small_chunkinputmetadata = ChunkInputMetadata(prompt_lens = [len(small_input_tokens_ids)],
                                                      kv_prefixs = [0],
                                                      kv_prefixs_blocks = None,
                                                      kv_block = None,
                                                      idxs = [len(small_input_tokens_ids) - 1],
                                                      sampling_params_for_sampler = None,
                                                      do_cat = False)

        test_encoded = predict_tokenizer(input_prompt, 
                                        padding = "max_length", 
                                        truncation = True, 
                                        return_tensors = "pt", 
                                        max_length = 2048)
        test_encoded = test_encoded.to('cuda:2')
        predictions = predict_model(input_ids = test_encoded['input_ids'], 
                                    attention_mask = test_encoded['attention_mask'])
        predicted_label = torch.argmax(predictions.logits, dim = 1).item()
        print(f"original predicted label is {predicted_label}")

        output_token_list, logprobs = chunkrunner_13b._run_workers("execute_model",
                                            inputs = input_tokens_ids_tensor,
                                            inputs_positions = input_positions_tensor,
                                            chunkmetadata = chunkinputmetadata)
        print(f"output_token_list: {output_token_list}")
        print(f"logprobs: {logprobs}")

        st = time.time()
        predict_labels = chunkrunner_125m._run_workers("execute_predict_model",
                                    inputs = small_input_tokens_ids_tensor,
                                    inputs_positions = small_input_positions_tensor,
                                    chunkmetadata = small_chunkinputmetadata)
        ed = time.time()
        print(f"predict costs {ed - st} seconds")
        print(f"predict_labels: {predict_labels}")

        thread_a = threading.Thread(target = execute_small_model, args = (small_input_tokens_ids_tensor, small_input_positions_tensor, small_chunkinputmetadata))
        thread_b = threading.Thread(target = execute_big_model, args = (input_tokens_ids_tensor, input_positions_tensor, chunkinputmetadata))

        thread_a.start()
        thread_b.start()

        thread_a.join()
        thread_b.join()'''