import os
import torch
import time
from vllm._C import ops

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

@torch.inference_mode()
def run_new_single_query_cached_kv_attention(v) -> None:
    num_kv_heads = 10
    scale = float(1.0 / (128 ** 0.5))
    block_size = 16
    x = 16 // torch.tensor([], dtype=torch.float16).element_size()
    layer_num = 1
    layer_stride = num_kv_heads * 128 * block_size
    head_stride = 128 * block_size

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
        for i in range(1):
            query = torch.empty(3, 40, 128, dtype=torch.float16, device='cuda').uniform_(-1e-3, 1e-3)

            key_caches = [torch.empty(40, 10, 128 // x, 16, x, dtype=torch.float16, device='cuda').uniform_(-1e-3, 1e-3) for _ in range(10)]
            key_caches_addresses = ops.tensor_for_caches_addresses(key_caches)
            value_caches = [torch.empty(40, 10, 128, 16, dtype=torch.float16, device='cuda').uniform_(-1e-3, 1e-3) for _ in range(10)]
            value_caches_addresses = ops.tensor_for_caches_addresses(value_caches)
            
            block_tables_tensor = torch.tensor([[0, 1, 2], [3, -1, -1], [4, 5, -1]], dtype=torch.int, device='cuda')
            context_lens_tensor = torch.tensor([48, 16, 32], dtype=torch.int, device='cuda')
            max_context_len = 48

            key_cache2 = torch.empty(10, 10, 128 // x, 16, x, dtype=torch.float16, device='cuda').uniform_(-1e-3, 1e-3)
            value_cache2 = torch.empty(10, 10, 128, 16, dtype=torch.float16, device='cuda').uniform_(-1e-3, 1e-3)

            for i in range(10):
                key_cache2[i, :, :, :, :] = key_caches[i][1, :, :, :, :]
                value_cache2[i, :, :, :] = value_caches[i][1, :, :, :]

            block_tables_tensor2 = torch.tensor([[0, 1, 2], [3, -1, -1], [4, 5, -1]], dtype=torch.int, device='cuda')
            context_lens_tensor2 = torch.tensor([48, 16, 32], dtype=torch.int, device='cuda')

            output1 = torch.empty(3, 40, 128, dtype=torch.float16, device='cuda')
            output2 = torch.empty(3, 40, 128, dtype=torch.float16, device='cuda')

            a1 = time.time()
            ops.paged_attention_v1_block(
                output1,
                query,
                key_caches_addresses,
                value_caches_addresses,
                num_kv_heads,
                scale,
                block_tables_tensor,
                context_lens_tensor,
                block_size,
                max_context_len,
                None,
                "auto",
                layer_num,
                layer_stride,
                head_stride
            )
            b1 = time.time()
            print("v1 block us ", (b1-a1)*1000000)

            a2 = time.time()
            ops.paged_attention_v1(
                output2,
                query,
                key_cache2,
                value_cache2,
                num_kv_heads,
                scale,
                block_tables_tensor2,
                context_lens_tensor2,
                block_size,
                max_context_len,
                None,
                "auto"
            )
            b2 = time.time()
            print("v1 block us ", (b2-a2)*1000000)

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
                key_caches_addresses,
                value_caches_addresses,
                num_kv_heads,
                scale,
                block_tables_tensor,
                context_lens_tensor,
                block_size,
                max_context_len,
                None,
                "auto",
                layer_num,
                layer_stride,
                head_stride
            )
            t2 = time.time()
            print("v2 block us ", (t2-t1)*1000000)
            
            output4 = torch.empty(3, 40, 128, dtype=torch.float16, device='cuda')
            tmp_output2 = torch.empty(3, 40, 512, 128, dtype=torch.float16, device='cuda')
            exp_sums2 = torch.empty(3, 40, 512, dtype=torch.float32, device='cuda')
            max_logits2 = torch.empty_like(exp_sums2, device='cuda')
            t3 = time.time()
            ops.paged_attention_v2(
                output4,
                exp_sums2,
                max_logits2,
                tmp_output2,
                query,
                key_cache2,
                value_cache2,
                num_kv_heads,
                scale,
                block_tables_tensor2,
                context_lens_tensor2,
                block_size,
                max_context_len,
                None,
                "auto"
            )
            t4 = time.time()
            print("v2 us ", (t3-t4)*1000000)

            is_close1 = torch.allclose(output1, output2, atol=1e-3, rtol=1e-5)
            print("Tolerant Errors for V2." if is_close1 else "Wrong Code in V2!")

            is_close2 = torch.allclose(output3, output4, atol=1e-3, rtol=1e-5)
            print("Tolerant Errors for V2." if is_close2 else "Wrong Code in V2!")
            
            # if torch.isnan(output4).any():
            #     print("output4 contains NaN values")

run_new_single_query_cached_kv_attention(1)