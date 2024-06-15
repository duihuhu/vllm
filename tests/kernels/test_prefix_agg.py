import torch
import random
from vllm._C import ops
from vllm.attention.ops.prefix_prefill import context_attention_fwd
from vllm.attention.ops.prefix_prefill_block import context_attention_block_fwd
import time

NUM_HEADS = 40
NUM_QUERIES_PER_KV = 4
HEAD_SIZE = 128
DTYPES = torch.float16
CUDA_DIVCE = "cuda:2"

@torch.inference_mode()
def test() -> None:
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    torch.set_default_device(CUDA_DIVCE)
    torch.cuda.set_device(CUDA_DIVCE)

    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    cache_size = 640
    block_size = 32
    max_block_per_request = 64
    subquery_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
    seq_lens = [a + b for a, b in zip(subquery_lens, ctx_lens)]
    num_kv_heads = NUM_HEADS // NUM_QUERIES_PER_KV

    num_tokens = sum(subquery_lens)
    query = torch.empty(num_tokens, NUM_HEADS, HEAD_SIZE, dtype=DTYPES)
    query.uniform_(-1e-3, 1e-3)
    output1 = torch.empty(num_tokens, NUM_HEADS, HEAD_SIZE, dtype=DTYPES)
    output2 = torch.empty(num_tokens, NUM_HEADS, HEAD_SIZE, dtype=DTYPES)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, HEAD_SIZE, dtype=DTYPES)
    kv.uniform_(-1e-3, 1e-3)
    key, value = kv.unbind(dim=1)

    k_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          HEAD_SIZE,
                          dtype=DTYPES)
    v_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          HEAD_SIZE,
                          dtype=DTYPES)
    k = torch.zeros(sum(subquery_lens), num_kv_heads, HEAD_SIZE, dtype=DTYPES)
    v = torch.zeros(sum(subquery_lens), num_kv_heads, HEAD_SIZE, dtype=DTYPES)
    values = torch.arange(0, cache_size, dtype=torch.long)
    values = values[torch.randperm(cache_size)]
    block_table = values[:BS * max_block_per_request].view(
        BS, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.long)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.long)
    b_start_loc = torch.cumsum(torch.tensor([0] + subquery_lens[:-1],
                                            dtype=torch.long),
                               dim=0)
    max_input_len = MAX_SEQ_LEN

    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1], dtype=torch.long), dim=0)

    for i in range(BS):
        for j in range(subquery_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] +
                                            j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] +
                                              b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads,
                         HEAD_SIZE)[start_slot:end_slot].copy_(
                             key[start_loc:end_loc])
            v_cache.view(-1, num_kv_heads,
                         HEAD_SIZE)[start_slot:end_slot].copy_(
                             value[start_loc:end_loc])
            cur_ctx += block_size
            block_id += 1
        
        k_cache = k_cache.view(-1, block_size, num_kv_heads, HEAD_SIZE // 8,
                           8).permute(0, 2, 3, 1, 4).contiguous()
        v_cache = v_cache.view(-1, block_size, num_kv_heads,
                           HEAD_SIZE).permute(0, 2, 3, 1).contiguous()
        
        k_caches = []
        v_caches = []
        for i in range(cache_size):
            k = torch.zeros(3,
                            block_size,
                            num_kv_heads,
                            HEAD_SIZE // 8,
                            8,
                            dtype=DTYPES)
            v = torch.zeros(3,
                            block_size,
                            num_kv_heads,
                            HEAD_SIZE,
                            dtype=DTYPES)
            k[1, :, :, :, :] = k_cache[i, :, :, :, :].clone()
            v[1, :, :, :] = v_cache[i, :, :, :].clone()
            k_caches.append(k)
            v_caches.append(v)
        k_addr = ops.tensor_for_caches_addresses(k_caches)
        v_addr = ops.tensor_for_caches_addresses(v_caches)

        context_attention_fwd(query, k, v, output1, k_cache, v_cache, block_table,
                          b_start_loc, b_seq_len, b_ctx_len, max_input_len, None)
        st1 = time.time()
        context_attention_fwd(query, k, v, output1, k_cache, v_cache, block_table,
                          b_start_loc, b_seq_len, b_ctx_len, max_input_len, None)
        ed1 = time.time()

        context_attention_block_fwd(1, block_size, num_kv_heads, HEAD_SIZE, 8, query, k,
                                    v, output2, k_addr, v_addr, block_table, b_start_loc,
                                    b_seq_len, b_ctx_len, max_input_len, None)
        st2 = time.time()
        context_attention_block_fwd(1, block_size, num_kv_heads, HEAD_SIZE, 8, query, k,
                                    v, output2, k_addr, v_addr, block_table, b_start_loc,
                                    b_seq_len, b_ctx_len, max_input_len, None)
        ed2 = time.time()

        is_close = torch.allclose(output1, output2, atol=1e-3, rtol=1e-5)
        if is_close:
            print(f"Pass")
        else:
            print(f"Error")
        
        print(f"Origion Costs {ed1 - st1} while Agg-Block Costs {ed2 - st2}")

test()