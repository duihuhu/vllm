import os

import torch

from vllm._C import ops

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

@torch.inference_mode()
def run_new_single_query_cached_kv_attention(v) -> None:
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 40 layers, 40 heads, 5120 dimension -> kv_heads = 10
    num_kv_heads = 10

    scale = float(1.0 / (128 ** 0.5))

    block_size = 16

    x = 16 // torch.tensor([], dtype = torch.float16).element_size()

    layer_num = 1
    if v == 1:
        query = torch.empty(3, 40, 128, dtype = torch.float16, device = 'cuda')
        query.uniform_(-1e-3, 1e-3)

        key_caches = []
        for _ in range(10):
            key_cache = torch.empty(size = (40, 10, 128 // x, 16, x),
                                    dtype = torch.float16,
                                    device = 'cuda')
            key_cache.uniform_(-1e-3, 1e-3)
            key_caches.append(key_cache)
        
        value_caches = []
        for _ in range(10):
            value_cache = torch.empty(size = (40, 10, 128, 16),
                                    dtype = torch.float16,
                                    device = 'cuda')
            value_cache.uniform_(-1e-3, 1e-3)
            value_caches.append(value_cache)
        
        # [[0, 1, -1], [2, -1, -1], [3, 4, 5]]
        block_tables_list = [[0, 1, 9], [2, 9, 9], [3, 4, 5]]
        block_tables_tensor = torch.tensor(block_tables_list, dtype = torch.int, device = 'cuda')

        context_lens_list = [32, 16, 48]
        context_lens_tensor = torch.tensor(context_lens_list, dtype = torch.int, device = 'cuda')

        max_context_len = 48

        output = torch.empty(3, 40, 128, dtype = torch.float16, device = 'cuda')

        ops.paged_attention_v1_block(
            output,
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

        output2 = torch.empty(3, 40, 128, dtype = torch.float16, device = 'cuda')

        key_cache2 = torch.empty(size = (10, 10, 128 // x, 16, x),
                                dtype = torch.float16,
                                device = 'cuda')
        key_cache2.uniform_(-1e-3, 1e-3)
        for i, key_cache_item in enumerate(key_caches):
            key_cache2[i, :, :, :, :] = key_cache_item[1, :, :, :, :].clone()

        value_cache2 = torch.empty(size = (10, 10, 128, 16),
                                dtype = torch.float16,
                                device = 'cuda')
        value_cache2.uniform_(-1e-3, 1e-3)
        for i, value_cache_item in enumerate(value_caches):
            value_cache2[i, :, :, :] = value_cache_item[1, :, :, :].clone()
        
        # In work.py when we really set the inputmetadata, we need a set () & list [] to set the block_tables
        block_tables_list2 = [[0, 1, 9], [2, 9, 9], [3, 4, 5]]
        block_tables_tensor2 = torch.tensor(block_tables_list2, dtype = torch.int, device = 'cuda')

        context_lens_list2 = [32, 16, 48]
        context_lens_tensor2 = torch.tensor(context_lens_list2, dtype = torch.int, device = 'cuda')

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

        is_close = torch.allclose(output, output2, atol = 1e-3, rtol = 1e-5)
        if is_close:
            print("Tolerant Errors for V1.")
        else:
            print("Wrong Code in V1!")
    else:
        query2 = torch.empty(1, 40, 128, dtype = torch.float16, device = 'cuda')
        query2.uniform_(-1e-3, 1e-3)
        key_caches3 = []
        for _ in range(64):
            key_cache = torch.empty(size = (40, 10, 128 // x, 16, x),
                                    dtype = torch.float16, 
                                    device = 'cuda')
            key_cache.uniform_(-1e-3, 1e-3)
            key_caches3.append(key_cache)
        
        value_caches3 = []
        for _ in range(64):
            value_cache = torch.empty(size = (40, 10, 128, 16),
                                    dtype = torch.float16,
                                    device = 'cuda')
            value_cache.uniform_(-1e-3, 1e-3)
            value_caches3.append(value_cache)
        block_tables_list3 = []
        temp = []
        for i in range(64):
            temp.append(i)
        block_tables_list3.append(temp)
        block_tables_tensor3 = torch.tensor(block_tables_list3, dtype = torch.int, device = 'cuda')
        context_lens_list3 = [1024]
        context_lens_tensor3 = torch.tensor(context_lens_list3, dtype = torch.int, device = 'cuda')
        max_context_len2 = 1024
        output3 = torch.empty(1, 40, 128, dtype = torch.float16, device = 'cuda')
        tmp_output1 = torch.empty(
                size = (1, 40, 512, 128),
                dtype = torch.float16)
        exp_sums1 = torch.empty(
                size = (1, 40, 512),
                dtype=torch.float32)
        max_logits1 = torch.empty_like(exp_sums1)
        ops.paged_attention_v2_block(
                output3,
                exp_sums1,
                max_logits1,
                tmp_output1,
                query2,
                key_caches3,
                value_caches3,
                num_kv_heads,
                scale,
                block_tables_tensor3,
                context_lens_tensor3,
                block_size,
                max_context_len2,
                None,
                "auto",
                layer_num
            )

        output4 = torch.empty(1, 40, 128, dtype = torch.float16, device = 'cuda')

        key_cache3 = torch.empty(size = (64, 10, 128 // x, 16, x),
                                dtype = torch.float16,
                                device = 'cuda')
        key_cache3.uniform_(-1e-3, 1e-3)
        for i, key_cache_item in enumerate(key_caches3):
            key_cache3[i, :, :, :, :] = key_cache_item[1, :, :, :, :].clone()

        value_cache3 = torch.empty(size = (64, 10, 128, 16),
                                dtype = torch.float16,
                                device = 'cuda')
        value_cache3.uniform_(-1e-3, 1e-3)
        for i, value_cache_item in enumerate(value_caches3):
            value_cache3[i, :, :, :] = value_cache_item[1, :, :, :].clone()
        
        # In work.py when we really set the inputmetadata, we need a set () & list [] to set the block_tables
        block_tables_list4 = []
        temp = []
        for i in range(64):
            temp.append(i)
        block_tables_list4.append(temp)
        block_tables_tensor4 = torch.tensor(block_tables_list4, dtype = torch.int, device = 'cuda')
        context_lens_list4 = [1024]
        context_lens_tensor4 = torch.tensor(context_lens_list4, dtype = torch.int, device = 'cuda')
        tmp_output2 = torch.empty(
                size = (1, 40, 512, 128),
                dtype = torch.float16)
        exp_sums2 = torch.empty(
                size = (1, 40, 512),
                dtype=torch.float32)
        max_logits2 = torch.empty_like(exp_sums2)
        ops.paged_attention_v2_block(
                output4,
                exp_sums2,
                max_logits2,
                tmp_output2,
                query2,
                key_cache3,
                value_cache3,
                num_kv_heads,
                scale,
                block_tables_tensor4,
                context_lens_tensor4,
                block_size,
                max_context_len2,
                None,
                "auto"
            )
        
        is_close = torch.allclose(output3, output4, atol = 1e-3, rtol = 1e-5)
        if is_close:
            print("Tolerant Errors for V2.")
        else:
            print("Wrong Code in V2!")

run_new_single_query_cached_kv_attention(2)