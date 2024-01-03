import torch
import multiprocessing

from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.chunked.chunkrunner import ChunkRunner
from vllm.chunked.chunk import ChunkInputMetadata, ChunkSamplingParams

tokenizer = get_tokenizer("/workspace/models/facebook/opt-125m")
chunkrunner = ChunkRunner(tokenizer = tokenizer,
                          chunk_size = 512,
                          chunk_num = 10)
chunkrunner.set_self_configs(model = "/workspace/models/facebook/opt-125m",
                             predict_model = "/workspace/models/facebook/opt-125m",
                             tensor_parallel_size = 2)
chunkrunner.set_parallel_chunkworkers()

chunkinputmetadata = ChunkInputMetadata(prompt_lens = [512],
                                        kv_prefixs = [0],
                                        kv_prefixs_blocks = None,
                                        kv_block = None,
                                        idxs = [511],
                                        sampling_params_for_sampler = [ChunkSamplingParams(temperature = 0.0,
                                                                                           top_p = 1.0,
                                                                                           top_k = -1)],
                                        do_cat = False)
input_tokens_ids = [1] * 512
input_positions = list(range(512))
input_tokens_ids_tensor = torch.cuda.LongTensor(input_tokens_ids)
input_positions_tensor = torch.cuda.LongTensor(input_positions)

output_token_list, logprobs = chunkrunner._run_workers("execute_model",
                                        inputs = input_tokens_ids_tensor,
                                        inputs_positions = input_positions_tensor,
                                        chunkmetadata = chunkinputmetadata)
predict_labels = chunkrunner._run_workers("execute_predict_model",
                              inputs = input_tokens_ids_tensor,
                              inputs_positions = input_positions_tensor,
                              chunkmetadata = chunkinputmetadata)

print(f"output_token_list: {output_token_list}")
print(f"logprobs: {logprobs}")
print(f"predict_labels: {predict_labels}")