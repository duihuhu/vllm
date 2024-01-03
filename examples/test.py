import torch
import threading
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.chunked.chunkrunner import ChunkRunner
from vllm.chunked.chunk import ChunkInputMetadata, ChunkSamplingParams
from vllm.worker.worker import _pad_to_alignment

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

def execute_big_model(input_tokens_ids_tensor, input_positions_tensor, chunkinputmetadata):
    output_token_list, logprobs = chunkrunner_13b._run_workers("execute_model",
                                            inputs = input_tokens_ids_tensor,
                                            inputs_positions = input_positions_tensor,
                                            chunkmetadata = chunkinputmetadata)
    print(f"output_token_list: {output_token_list}")
    print(f"logprobs: {logprobs}")

def execute_small_model(input_tokens_ids_tensor, input_positions_tensor, chunkinputmetadata):
    predict_labels = chunkrunner_125m._run_workers("execute_predict_model",
                                inputs = input_tokens_ids_tensor,
                                inputs_positions = input_positions_tensor,
                                chunkmetadata = chunkinputmetadata)
    print(f"predict_labels: {predict_labels}")

if __name__ == "__main__":

    #tokenizer_13b = get_tokenizer("/workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/")
    tokenizer_13b = get_tokenizer("/workspace/models/facebook/opt-125m")
    chunkrunner_13b = ChunkRunner(tokenizer = tokenizer_13b,
                              chunk_size = 512,
                              chunk_num = 10)
    chunkrunner_13b.set_self_configs(model = "/workspace/models/facebook/opt-125m",
        #model = "/workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/",
                                 predict_model = None,
                                 tensor_parallel_size = 2)
    chunkrunner_13b.set_parallel_chunkworkers(num_gpus = 0.7)

    #tokenizer_125m = get_tokenizer("/workspace/opt-125m")
    tokenizer_125m = get_tokenizer("/workspace/models/facebook/opt-125m")
    chunkrunner_125m = ChunkRunner(tokenizer = tokenizer_125m,
                              chunk_size = 512,
                              chunk_num = 10)
    chunkrunner_125m.set_self_configs(model = None,
                                 predict_model = "/workspace/opt_125m_model_sharegpt",
                                 tensor_parallel_size = 2)
    chunkrunner_125m.set_parallel_chunkworkers(num_gpus = 0.3)


    #predict_tokenizer = AutoTokenizer.from_pretrained("/workspace/opt-125m")
    predict_tokenizer = AutoTokenizer.from_pretrained("/workspace/models/facebook/opt-125m")
    predict_model = AutoModelForSequenceClassification.from_pretrained("/workspace/opt_125m_model_sharegpt", num_labels = 10)
    predict_model = predict_model.to('cuda:2')
    
    chunkinputmetadata = ChunkInputMetadata(prompt_lens = [512],
                                            kv_prefixs = [0],
                                            kv_prefixs_blocks = None,
                                            kv_block = None,
                                            idxs = [511],
                                            sampling_params_for_sampler = [ChunkSamplingParams(temperature = 0.0,
                                                                                            top_p = 1.0,
                                                                                            top_k = -1)],
                                            do_cat = False)
    filtered_dataset = set_inputs(tokenizer = tokenizer_13b,
                                  #dataset_path = "/workspace/ShareGPT_V3_unfiltered_cleaned_split.json",
                                  dataset_path = "/workspace/datasets/ShareGPT_V3_unfiltered_cleaned_split.json",
                                  num_requests = 32)
    
    for input_prompt, input_tokens_ids, output_len in filtered_dataset:
        print(f"output_len is {output_len}")

        small_input_tokens_ids = input_tokens_ids
        small_input_tokens_ids = _pad_to_alignment(small_input_tokens_ids, multiple_of = 8)
      
        input_tokens_ids = _pad_to_alignment(input_tokens_ids, multiple_of = 8)
        input_positions = list(range(len(input_tokens_ids)))
        input_tokens_ids_tensor = torch.cuda.LongTensor(input_tokens_ids)
        input_positions_tensor = torch.cuda.LongTensor(input_positions)
        chunkinputmetadata = ChunkInputMetadata(prompt_lens = [len(input_tokens_ids)],
                                            kv_prefixs = [0],
                                            kv_prefixs_blocks = None,
                                            kv_block = None,
                                            idxs = [len(input_tokens_ids) - 1],
                                            sampling_params_for_sampler = [ChunkSamplingParams(temperature = 0.0,
                                                                                            top_p = 1.0,
                                                                                            top_k = -1)],
                                            do_cat = False)

        small_input_positions = list(range(len(small_input_tokens_ids)))
        small_input_tokens_ids_tensor = torch.cuda.LongTensor(small_input_tokens_ids)
        small_input_positions_tensor = torch.cuad.LongTensor(small_input_positions)
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

        thread_a = threading.Thread(target = execute_small_model, args = (small_input_tokens_ids_tensor, small_input_positions_tensor, small_chunkinputmetadata))
        thread_b = threading.Thread(target = execute_big_model, args = (input_tokens_ids_tensor, input_positions_tensor, chunkinputmetadata))

        thread_a.start()
        thread_b.start()

        thread_a.join()
        thread_b.join()    