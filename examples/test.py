import torch
import threading
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.chunked.chunkrunner import ChunkRunner
from vllm.chunked.chunk import ChunkInputMetadata, ChunkSamplingParams
from vllm.worker.worker import _pad_to_max

def execute_big_model():
    output_token_list, logprobs = chunkrunner._run_workers("execute_model",
                                            inputs = input_tokens_ids_tensor,
                                            inputs_positions = input_positions_tensor,
                                            chunkmetadata = chunkinputmetadata)
    print(f"output_token_list: {output_token_list}")
    print(f"logprobs: {logprobs}")

def execute_small_model():
    predict_labels = chunkrunner._run_workers("execute_predict_model",
                                inputs = input_tokens_ids_tensor,
                                inputs_positions = input_positions_tensor,
                                chunkmetadata = chunkinputmetadata)
    print(f"predict_labels: {predict_labels}")

if __name__ == "__main__":

    tokenizer = get_tokenizer("/workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/")
    chunkrunner = ChunkRunner(tokenizer = tokenizer,
                              chunk_size = 512,
                              chunk_num = 10)
    chunkrunner.set_self_configs(model = "/workspace/opt-13b/model/snapshots/e515202d1e7750da62d245fbccb2723b9c1790f5/",
                                 predict_model = "/workspace/opt_125m_model_sharegpt",
                                 tensor_parallel_size = 2)
    chunkrunner.set_parallel_chunkworkers()

    predict_tokenizer = AutoTokenizer.from_pretrained("/workspace/opt-125m")
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
    input_prompt = "test whether opt-13b and opt-125m can be co-runnging"
    input_tokens_ids = tokenizer(input_prompt).input_ids
    if len(input_tokens_ids) < 512:
        input_tokens_ids = _pad_to_max(input_tokens_ids, max_len = 512)
    else:
        input_tokens_ids = input_tokens_ids[0: 512]
    input_positions = list(range(512))
    input_tokens_ids_tensor = torch.cuda.LongTensor(input_tokens_ids)
    input_positions_tensor = torch.cuda.LongTensor(input_positions)

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

    thread_a = threading.Thread(target = execute_small_model)
    thread_b = threading.Thread(target = execute_big_model)

    thread_a.start()
    thread_b.start()

    thread_a.join()
    thread_b.join()


    