/*
 * Adapted from https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "attention_dtypes.h"
#include "attention_utils.cuh"

#include <algorithm>
#include <vector>

#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace vllm {

// Utility function for attention softmax.
template<int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Broadcast to other threads.
  return __shfl_sync(uint32_t(-1), sum, 0);
}

// Grid: (num_heads, num_seqs).
template<
  typename scalar_t,
  int HEAD_SIZE,
  int BLOCK_SIZE,
  int NUM_THREADS>
__global__ void new_single_query_cached_kv_attention_kernel(
  scalar_t* __restrict__ out,             // [num_seqs, num_heads, head_size]
  const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
  //const scalar_t* __restrict__ k_cache,   // [num_blocks, num_heads, head_size/x, block_size, x] -> [num_blocks]
  //const scalar_t* __restrict__ v_cache,   // [num_blocks, num_heads, head_size, block_size]
  int64_t* key_cache_ptrs, // 记得扣除重复的
  int64_t* value_cache_ptrs,
  const float scale,
  const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq] 
  // [[0,1,-1],[2,3,4],[5,-1,-1]]
  const int* __restrict__ context_lens,   // [num_seqs] -> 其实就是每个q对应的kv的长度
  const int max_num_blocks_per_seq, // 用于在block_tables里面遍历，锁定到对应seq_id的那段blocktables
  // 3
  const float* __restrict__ alibi_slopes, // [num_heads]
  const int q_stride, // Q的第一个维度
  const int layer_num) { 

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1); // 一个warp处理一个block里面的token，蕴含有blocksize个threadgroup
  constexpr int NUM_TOKENS_PER_THREAD_GROUP = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE; // 蕴含着一个threadblock处理一个token
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int seq_idx = blockIdx.y; // 因为cuda的坐标是y轴向下的，所以每一行刚好就是一个q token <-> 也确实对应一个seq
  const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread group
  // fetch or compute 16 bytes at a time.
  // For example, if the size of a thread group is 4 and the data type is half,
  // then the vector size is 16 / (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1); // 一个threadgroup每次处理16个字节的元素 -> vecsize代表，这16个字节的数据是几个元素，其实也蕴含一个thread一个元素
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE; // 一个thread要处理多少个元素 -> 因为已经蕴含过一个threadgroup处理一个token，又因为一个threadblock只处理一个头，所以总元素大小是headsize
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE; // 一个thread需要循环多少次 -> 因为一次只处理vecsize个元素

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in the group
  // has 0, 4, 8, ... th vectors of the query, and the second thread has 1, 5, 9, ...
  // th vectors of the query, and so on.
  // NOTE(woosuk): Because q is split from a qkv tensor, it may not be contiguous.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE; // 因为Q这个tensor的shape是[num_seqs, num_heads, head_size]
  Q_vec q_vecs[NUM_VECS_PER_THREAD]; // 每个thread要读取的数据都放到这个二维vector里
#pragma unroll 
  for (int i = 0; i < NUM_VECS_PER_THREAD; i++) { // 这行代码说明读数据是深入到bit位的
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE; // 每次读取q的起始地址
    q_vecs[i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE); // 每轮跨过的是一个threadgroup的处理空间，所以因该是+一个vecsize，代表处理了这么多个元素
  }

  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(scalar_t); // K这里，任然是一个threadgroup处理16个字节，x代表的是这16个字节对应多少个元素
  float qk_max = -FLT_MAX; // 因为需要规约，需要调整分子/分母

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq; // 找到blocktable的起始点
  const int context_len = context_lens[seq_idx]; // 这个seq参与计算的kv有多长
  const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE; // 计算这个seq到底有多少个block的kv cache

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  // 这一块和q一样，一个warp处理一个blocksize里的所有k token，一个threadgroup处理一个token
  for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) { // 需要遍历几次，就这样算，q因为是连续的所以直接下滑到thread级别来处理
    // block_table应该改成储存addr的list
    const int physical_block_number = block_table[block_idx]; // 第几个block（0，1，2...）-> 真实的block id
    
    const scalar_t* __restrict__ k_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[physical_block_number]);
    
    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the the thread group size is 4, then the first thread in the group
    // has 0, 4, 8, ... th vectors of the key, and the second thread has 1, 5, 9, ... th
    // vectors of the key, and so on.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE; // 这个threadgroup，处理这个block里面的哪一个k token
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        // 下面第一行，k_cache + physical_block_number * num_heads * HEAD_SIZE * BLOCK_SIZE，改为一个address输入
        const scalar_t* k_ptr = k_cache + layer_num * num_heads * HEAD_SIZE * BLOCK_SIZE //physical_block_number * num_heads * HEAD_SIZE * BLOCK_SIZE 
                                        + head_idx * HEAD_SIZE * BLOCK_SIZE
                                        + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;
        k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs, k_vecs);
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - context_len) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= context_len;
        logits[token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD = (HEAD_SIZE + NUM_ROWS_PER_ITER - 1) / NUM_ROWS_PER_ITER;

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
    const int physical_block_number = block_table[block_idx];

    const scalar_t* __restrict__ v_cache = reinterpret_cast<scalar_t*>(value_cache_ptrs[physical_block_number]);

    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx));
    // 同理，v_cache + physical_block_number * num_heads * HEAD_SIZE * BLOCK_SIZE应该改成一个addr的入口
    const scalar_t* v_ptr = v_cache + 
    layer_num * num_heads * HEAD_SIZE * BLOCK_SIZE + 
    head_idx * HEAD_SIZE * BLOCK_SIZE;
    
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += __shfl_xor_sync(uint32_t(-1), acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for logits
  // is reused for the output.
  __syncthreads();

  // Perform reduction across warps.
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    __syncthreads();
  }

  // Write the final output.
  if (warp_idx == 0) {
    scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}

} // namespace vllm

#define LAUNCH_ATTENTION_KERNEL(T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS)                        \
  vllm::new_single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>        \
  <<<grid, block, shared_mem_size, stream>>>(                                                 \
    out_ptr,                                                                                  \
    query_ptr,                                                                                \
    key_cache_ptrs,                                                                            \
    value_cache_ptrs,                                                                          \
    scale,                                                                                    \
    block_tables_ptr,                                                                         \
    context_lens_ptr,                                                                         \
    max_num_blocks_per_seq,                                                                   \
    alibi_slopes_ptr,                                                                         \
    query_stride,                                                                             \
    layer_num);

// TODO(woosuk): Tune NUM_THREADS.
template<
  typename T,
  int BLOCK_SIZE,
  int NUM_THREADS = 128>
void new_single_query_cached_kv_attention_launcher( // 本质上按照，元素类型、头的个数、blocksize大小来启动kernel，启动顺序依次从外至内
  torch::Tensor& out,
  torch::Tensor& query,
  //torch::Tensor& key_cache,
  //torch::Tensor& value_cache,
  std::vector<torch::Tensor>& key_caches,
  std::vector<torch::Tensor>& value_caches,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes,
  
  int layer_num) {
  
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int query_stride = query.stride(0);

  int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr = alibi_slopes ?
    reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
    : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  //T* key_cache_ptr = reinterpret_cast<T*>(key_cache.data_ptr());
  //T* value_cache_ptr = reinterpret_cast<T*>(value_cache.data_ptr());

  torch::Device cache_device = key_caches[0].device();
  TORCH_CHECK(cache_device.is_cuda());
  
  int total_blocks_num = key_caches.size();
  
  int64_t key_cache_ptrs_array[total_blocks_num];
  int64_t value_cache_ptrs_array[total_blocks_num];
  
  for (int i = 0; i < total_blocks_num; i++) {
    key_cache_ptrs_array[i] = reinterpret_cast<int64_t>(key_caches[i].data_ptr());
    value_cache_ptrs_array[i] = reinterpret_cast<int64_t>(value_caches[i].data_ptr());
  }
  
  torch::Tensor key_cache_ptrs_tensor = torch::from_blob(
    key_cache_ptrs_array, {total_blocks_num}, torch::kInt64).to(cache_device);
  torch::Tensor value_cache_ptrs_tensor = torch::from_blob(
    value_cache_ptrs_array, {total_blocks_num}, torch::kInt64).to(cache_device);
  
  int64_t* key_cache_ptrs = key_cache_ptrs_tensor.data_ptr<int64_t>();    
  int64_t* value_cache_ptrs = value_cache_ptrs_tensor.data_ptr<int64_t>(); 

  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* context_lens_ptr = context_lens.data_ptr<int>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_context_len = ((max_context_len + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_context_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs); // 一个threadblock处理一个seq的一个头
  dim3 block(NUM_THREADS); // 一个threadblock有128个线程
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we omitted head sizes
    // 32, 160, 192.
    // case 32:
    //   LAUNCH_ATTENTION_KERNEL(T, 32, BLOCK_SIZE, NUM_THREADS);
    //   break;
    case 64:
      LAUNCH_ATTENTION_KERNEL(T, 64, BLOCK_SIZE, NUM_THREADS);
      break;
    case 80:
      LAUNCH_ATTENTION_KERNEL(T, 80, BLOCK_SIZE, NUM_THREADS);
      break;
    case 96:
      LAUNCH_ATTENTION_KERNEL(T, 96, BLOCK_SIZE, NUM_THREADS);
      break;
    case 112:
      LAUNCH_ATTENTION_KERNEL(T, 112, BLOCK_SIZE, NUM_THREADS);
      break;
    case 128:
      LAUNCH_ATTENTION_KERNEL(T, 128, BLOCK_SIZE, NUM_THREADS);
      break;
    // case 160:
    //   LAUNCH_ATTENTION_KERNEL(T, 160, BLOCK_SIZE, NUM_THREADS);
    //   break;
    // case 192:
    //   LAUNCH_ATTENTION_KERNEL(T, 192, BLOCK_SIZE, NUM_THREADS);
    //   break;
    case 256:
      LAUNCH_ATTENTION_KERNEL(T, 256, BLOCK_SIZE, NUM_THREADS);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
}

#define CALL_KERNEL_LAUNCHER(T, BLOCK_SIZE)                         \
  new_single_query_cached_kv_attention_launcher<T, BLOCK_SIZE>(         \
    out,                                                            \
    query,                                                          \
    key_caches,                                                      \
    value_caches,                                                    \
    scale,                                                          \
    block_tables,                                                   \
    context_lens,                                                   \
    max_context_len,                                                \
    alibi_slopes,                                                   \
    layer_num);

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_KERNEL_LAUNCHER_BLOCK_SIZE(T)                          \
  switch (block_size) {                                             \
    /* case 1:                         */                           \
    /*   CALL_KERNEL_LAUNCHER(T, 1);   */                           \
    /*   break;                        */                           \
    /* case 2:                         */                           \
    /*   CALL_KERNEL_LAUNCHER(T, 2);   */                           \
    /*   break;                        */                           \
    /* case 4:                         */                           \
    /*   CALL_KERNEL_LAUNCHER(T, 4);   */                           \
    /*   break;                        */                           \
    case 8:                                                         \
      CALL_KERNEL_LAUNCHER(T, 8);                                   \
      break;                                                        \
    case 16:                                                        \
      CALL_KERNEL_LAUNCHER(T, 16);                                  \
      break;                                                        \
    case 32:                                                        \
      CALL_KERNEL_LAUNCHER(T, 32);                                  \
      break;                                                        \
    /* case 64:                        */                           \
    /*   CALL_KERNEL_LAUNCHER(T, 64);  */                           \
    /*   break;                        */                           \
    /* case 128:                       */                           \
    /*   CALL_KERNEL_LAUNCHER(T, 128); */                           \
    /*   break;                        */                           \
    /* case 256:                       */                           \
    /*   CALL_KERNEL_LAUNCHER(T, 256); */                           \
    /*   break;                        */                           \
    default:                                                        \
      TORCH_CHECK(false, "Unsupported block size: ", block_size);   \
      break;                                                        \
  }

void new_single_query_cached_kv_attention(
  torch::Tensor& out,             // [num_seqs, num_heads, head_size]
  torch::Tensor& query,           // [num_seqs, num_heads, head_size]
  //torch::Tensor& key_cache,       // [num_blocks, num_heads, head_size/x, block_size, x]
  //torch::Tensor& value_cache,     // [num_blocks, num_heads, head_size, block_size]
  std::vector<torch::Tensor>& key_caches, // [num_layers, num_heads, head_size/x, block_size, x]
  std::vector<torch::Tensor>& value_caches, // [num_layers, num_heads, head_size, block_size]
  float scale,
  torch::Tensor& block_tables,    // [num_seqs, max_num_blocks_per_seq]
  torch::Tensor& context_lens,    // [num_seqs]
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes,
  int layer_num) {
  if (query.dtype() == at::ScalarType::Float) {
    CALL_KERNEL_LAUNCHER_BLOCK_SIZE(float);
  } else if (query.dtype() == at::ScalarType::Half) {
    CALL_KERNEL_LAUNCHER_BLOCK_SIZE(uint16_t);
  } else if (query.dtype() == at::ScalarType::BFloat16) {
    CALL_KERNEL_LAUNCHER_BLOCK_SIZE(__nv_bfloat16);
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
  }
}

#undef WARP_SIZE
#undef MAX
#undef MIN