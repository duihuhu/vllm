import os
import torch
import time
from vllm._C import ops

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

@torch.inference_mode()
def run_new_single_query_cached_kv_attention(v) -> None:
    num_kv_heads = 10
    scale = float(1.0 / (128 ** 0.5))
    block_size = 16
    x = 16 // torch.tensor([], dtype=torch.float16).element_size()
    layer_num = 1

    def print_cuda_memory():
        print(f"Allocated: {torch.cuda.memory_allocated()} bytes")
        print(f"Cached: {torch.cuda.memory_reserved()} bytes")

    def check_tensor_device(tensor, name):
        if not tensor.is_cuda:
            print(f"{name} is not on CUDA")
        else:
            print(f"{name} is on {tensor.device}")

    def check_for_invalid_values(tensor, name):
        if torch.isnan(tensor).any():
            print(f"{name} contains NaN values")
        if torch.isinf(tensor).any():
            print(f"{name} contains Inf values")

    if v == 1:
        torch.cuda.empty_cache()
        print_cuda_memory()
        for i in range(10):
            query = torch.empty(3, 40, 128, dtype=torch.float16, device='cuda').uniform_(-1e-3, 1e-3)

            key_caches = [torch.empty(40, 10, 128 // x, 16, x, dtype=torch.float16, device='cuda').uniform_(-1e-3, 1e-3) for _ in range(10)]

            value_caches = [torch.empty(40, 10, 128, 16, dtype=torch.float16, device='cuda').uniform_(-1e-3, 1e-3) for _ in range(10)]
            block_tables_tensor = torch.tensor([[0, 1, 9], [2, 9, 9], [3, 4, 5]], dtype=torch.int, device='cuda')
            context_lens_tensor = torch.tensor([32, 16, 48], dtype=torch.int, device='cuda')
            max_context_len = 48

            key_cache2 = torch.empty(10, 10, 128 // x, 16, x, dtype=torch.float16, device='cuda').uniform_(-1e-3, 1e-3)
            value_cache2 = torch.empty(10, 10, 128, 16, dtype=torch.float16, device='cuda').uniform_(-1e-3, 1e-3)

            for i in range(10):
                key_cache2[i, :, :, :, :] = key_caches[i][1, :, :, :, :]
                value_cache2[i, :, :, :] = value_caches[i][1, :, :, :]

            block_tables_tensor2 = torch.tensor([[0, 1, 9], [2, 9, 9], [3, 4, 5]], dtype=torch.int, device='cuda')
            context_lens_tensor2 = torch.tensor([32, 16, 48], dtype=torch.int, device='cuda')

            # print("Tolerant Errors for V1." if is_close else "Wrong Code in V1!")

            output3 = torch.empty(3, 40, 128, dtype=torch.float16, device='cuda')
            tmp_output1 = torch.empty(3, 40, 512, 128, dtype=torch.float16, device='cuda')
            exp_sums1 = torch.empty(3, 40, 512, dtype=torch.float32, device='cuda')
            max_logits1 = torch.empty_like(exp_sums1, device='cuda')
            t1 = time.time()
            ops.paged_attention_v2_block(
                output3,
                exp_sums1,
                max_logits1,
                tmp_output1,
                query,
                key_caches,
                value_caches,
                num_kv_heads,
                scale,
                block_tables_tensor,
                context_lens_tensor,
                block_size,
                max_context_len,
                None,
                "auto",
                layer_num
            )
            t2 = time.time()
            print("v2 block ", t2-t1)
            
            # output4 = torch.empty(3, 40, 128, dtype=torch.float16, device='cuda')
            # tmp_output2 = torch.empty(3, 40, 512, 128, dtype=torch.float16, device='cuda')
            # exp_sums2 = torch.empty(3, 40, 512, dtype=torch.float32, device='cuda')
            # max_logits2 = torch.empty_like(exp_sums2, device='cuda')
            # t1 = time.time()
            # ops.paged_attention_v2(
            #     output4,
            #     exp_sums2,
            #     max_logits2,
            #     tmp_output2,
            #     query,
            #     key_cache2,
            #     value_cache2,
            #     num_kv_heads,
            #     scale,
            #     block_tables_tensor2,
            #     context_lens_tensor2,
            #     block_size,
            #     max_context_len,
            #     None,
            #     "auto"
            # )
            # t2 = time.time()
            # print("v2 ", t2-t1)
            # is_close = torch.allclose(output3, output4, atol=1e-3, rtol=1e-5)
            # print("Tolerant Errors for V2." if is_close else "Wrong Code in V2!")
            
            # if torch.isnan(output4).any():
            #     print("output4 contains NaN values")

run_new_single_query_cached_kv_attention(1)