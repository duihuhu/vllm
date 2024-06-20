#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"
#ifdef ENABLE_FP8_E5M2
#include "quantization/fp8_e5m2_kvcache/quant_utils.cuh"
#endif

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
  typedef __hip_bfloat16 __nv_bfloat16;
#endif

void swap_agg_block(
  torch::Tensor& src,
  torch::Tensor& dst,
  const int64_t block_size_in_bytes) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  cudaMemcpyKind memcpy_type;
  if (src_device.is_cuda() && dst_device.is_cuda()) {
    TORCH_CHECK(
      src_device.index() == dst_device.index(),
      "src and dst must be on the same GPU");
    memcpy_type = cudaMemcpyDeviceToDevice;
  } else if (src_device.is_cuda() && dst_device.is_cpu()) {
    memcpy_type = cudaMemcpyDeviceToHost;
  } else if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = cudaMemcpyHostToDevice;
  } else {
    TORCH_CHECK(false, "Invalid device combination");
  }

  char *src_ptr = static_cast<char*>(src.data_ptr());
  char *dst_ptr = static_cast<char*>(dst.data_ptr());

  const at::cuda::OptionalCUDAGuard device_guard(src_device.is_cuda() ? src_device : dst_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  cudaMemcpyAsync(
      dst_ptr,
      src_ptr,
      block_size_in_bytes,
      memcpy_type,
      stream);
}

namespace vllm {

// Grid: (num_layers, num_pairs)
template<typename scalar_t>
__global__ void copy_blocks_agg_kernel(
  int64_t* key_cache_ptrs,
  int64_t* value_cache_ptrs,
  const int64_t* __restrict__ block_mapping,
  const int numel_per_layer) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  //scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  //scalar_t* value_cache = reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];
  scalar_t* key_cache_src = reinterpret_cast<scalar_t*>(key_cache_ptrs[src_block_number]);
  scalar_t* value_cache_src = reinterpret_cast<scalar_t*>(value_cache_ptrs[src_block_number]);
  scalar_t* key_cache_dst = reinterpret_cast<scalar_t*>(key_cache_ptrs[dst_block_number]);
  scalar_t* value_cache_dst = reinterpret_cast<scalar_t*>(value_cache_ptrs[dst_block_number]);
 
  const int64_t layer_offset = layer_idx * numel_per_layer;
  //const int64_t dst_block_offset = dst_block_number * numel_per_layer;
  for (int i = threadIdx.x; i < numel_per_layer; i += blockDim.x) {
    int64_t thread_offset = layer_offset + i;
    //int64_t dst_offset = layer_offset + i;
    key_cache_dst[thread_offset] = key_cache_src[thread_offset];
  }
  for (int i = threadIdx.x; i < numel_per_layer; i += blockDim.x) {
    int64_t thread_offset = layer_offset + i;
    //int64_t src_offset = src_block_offset + i;
    //int64_t dst_offset = dst_block_offset + i;
    value_cache_dst[thread_offset] = value_cache_src[thread_offset];
  }
}

} // namespace vllm

void copy_blocks_agg(
  torch::Tensor& key_caches_addresses, // Add by hhy: change input from data tensors to data pointers
  torch::Tensor& value_caches_addresses,
  torch::Tensor& data_type_tensor,
  const std::map<int64_t, std::vector<int64_t>>& block_mapping,
  const int num_layers,
  const int numel_per_layer) {
  int num_blocks = key_caches_addresses.size(0);
  TORCH_CHECK(num_blocks == value_caches_addresses.size(0));
  if (num_blocks == 0) {
    return;
  }
  torch::Device cache_device = key_caches_addresses[0].device();
  TORCH_CHECK(cache_device.is_cuda());

  // Add by hhy: Already pointers!
  // Create data structures for the kernel.
  // Create an array of pointers to the key and value caches.
  /*int64_t key_cache_ptrs[num_layers];
  int64_t value_cache_ptrs[num_layers];
  for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    key_cache_ptrs[layer_idx] = reinterpret_cast<int64_t>(key_caches[layer_idx].data_ptr());
    value_cache_ptrs[layer_idx] = reinterpret_cast<int64_t>(value_caches[layer_idx].data_ptr());
  }*/

  // Create block mapping array.
  std::vector<int64_t> block_mapping_vec;
  for (const auto& pair : block_mapping) {
    int64_t src_block_number = pair.first;
    for (int64_t dst_block_number : pair.second) {
      block_mapping_vec.push_back(src_block_number);
      block_mapping_vec.push_back(dst_block_number);
    }
  }
  int64_t* block_mapping_array = block_mapping_vec.data();
  int num_pairs = block_mapping_vec.size() / 2;

  // Add by hhy: Already pointers!
  // Move the data structures to the GPU.
  // NOTE: This synchronizes the CPU and GPU.
  /*torch::Tensor key_cache_ptrs_tensor = torch::from_blob(
    key_cache_ptrs, {num_layers}, torch::kInt64).to(cache_device);
  torch::Tensor value_cache_ptrs_tensor = torch::from_blob(
    value_cache_ptrs, {num_layers}, torch::kInt64).to(cache_device);*/
  torch::Tensor block_mapping_tensor = torch::from_blob(
    block_mapping_array, {2 * num_pairs}, torch::kInt64).to(cache_device);

  // Launch the kernel.
  //const int numel_per_block = key_caches[0][0].numel();
  dim3 grid(num_layers, num_pairs);
  dim3 block(std::min(1024, numel_per_layer));
  const at::cuda::OptionalCUDAGuard device_guard(cache_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_AND_BYTE_TYPES(
    data_type_tensor.scalar_type(), "copy_blocks_agg_kernel", ([&] {
      vllm::copy_blocks_agg_kernel<scalar_t><<<grid, block, 0, stream>>>(
        key_caches_addresses.data_ptr<int64_t>(), //key_cache_ptrs_tensor.data_ptr<int64_t>(),
        value_caches_addresses.data_ptr<int64_t>(), //value_cache_ptrs_tensor.data_ptr<int64_t>(),
        block_mapping_tensor.data_ptr<int64_t>(),
        numel_per_layer);
    }));
}

namespace vllm {

template<typename scalar_t, typename cache_t, bool is_fp8_e5m2_kv_cache>
__global__ void reshape_and_cache_agg_kernel(
  const scalar_t* __restrict__ key,           // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ value,         // [num_tokens, num_heads, head_size]
  int64_t* __restrict__ key_cache_ptrs,       // key_cache_ptrs[i] == address of block i -> [num_layers, num_kv_heads, head_size // x, block_size, x]
  int64_t* __restrict__ value_cache_ptrs,
  //cache_t* __restrict__ key_cache,            // [num_blocks, num_heads, head_size/x, block_size, x]
  //cache_t* __restrict__ value_cache,          // [num_blocks, num_heads, head_size, block_size]
  const int64_t* __restrict__ slot_mapping,   // [num_tokens]
  const int key_stride,
  const int value_stride,
  const int num_heads,
  const int head_size,
  const int block_size,
  const int x,
  const int num_layer) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    cache_t* __restrict__ key_cache = reinterpret_cast<cache_t*>(key_cache_ptrs[block_idx]);
    const int64_t tgt_key_idx_ptr = num_layer * num_heads * (head_size / x) * block_size * x //block_idx * num_heads * (head_size / x) * block_size * x
                                + head_idx * (head_size / x) * block_size * x
                                + x_idx * block_size * x
                                + block_offset * x
                                + x_offset;
    cache_t* __restrict__ value_cache = reinterpret_cast<cache_t*>(value_cache_ptrs[block_idx]);
    const int64_t tgt_value_idx_ptr = num_layer * num_heads * head_size * block_size //block_idx * num_heads * head_size * block_size
                                  + head_idx * head_size * block_size
                                  + head_offset * block_size
                                  + block_offset;
    scalar_t tgt_key = key[src_key_idx];
    scalar_t tgt_value = value[src_value_idx];
    if constexpr (is_fp8_e5m2_kv_cache) {
#ifdef ENABLE_FP8_E5M2
      //key_cache[tgt_key_idx] = fp8_e5m2_unscaled::vec_conversion<uint8_t, scalar_t>(tgt_key);
      //value_cache[tgt_value_idx] = fp8_e5m2_unscaled::vec_conversion<uint8_t, scalar_t>(tgt_value);
      key_cache[tgt_key_idx_ptr] = fp8_e5m2_unscaled::vec_conversion<uint8_t, scalar_t>(tgt_key);
      value_cache[tgt_value_idx_ptr] = fp8_e5m2_unscaled::vec_conversion<uint8_t, scalar_t>(tgt_value);
#else
      assert(false);
#endif
    } else {
      //key_cache[tgt_key_idx] = tgt_key;
      //value_cache[tgt_value_idx] = tgt_value;
      key_cache[tgt_key_idx_ptr] = tgt_key;
      value_cache[tgt_value_idx_ptr] = tgt_value;
    }
  }
}

} // namespace vllm

#define CALL_RESHAPE_AND_CACHE_AGG(KV_T, CACHE_T, IS_FP8_E5M2_KV_CACHE)                                \
  vllm::reshape_and_cache_agg_kernel<KV_T, CACHE_T, IS_FP8_E5M2_KV_CACHE><<<grid, block, 0, stream>>>( \
    reinterpret_cast<KV_T*>(key.data_ptr()),                                                       \
    reinterpret_cast<KV_T*>(value.data_ptr()),                                                     \
    reinterpret_cast<int64_t*>(key_cache_addresses.data_ptr<int64_t>()),                                              \
    reinterpret_cast<int64_t*>(value_cache_addresses.data_ptr<int64_t>()),                                            \
    slot_mapping.data_ptr<int64_t>(),                                                              \
    key_stride,                                                                                    \
    value_stride,                                                                                  \
    num_heads,                                                                                     \
    head_size,                                                                                     \
    block_size,                                                                                    \
    x,                                                                                             \
    num_layer);

void reshape_and_cache_agg(
  torch::Tensor& key,           // [num_tokens, num_heads, head_size]
  torch::Tensor& value,         // [num_tokens, num_heads, head_size]
  torch::Tensor& key_cache_addresses,     // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache_addresses,   // [num_blocks, num_heads, head_size, block_size]
  torch::Tensor& slot_mapping,  // [num_tokens]
  const std::string& kv_cache_dtype,
  const int block_size,
  const int x,
  const int num_layer)
{
  int num_tokens = key.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  //int block_size = key_cache.size(3);
  //int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (kv_cache_dtype == "auto") {
    if (key.dtype() == at::ScalarType::Float) {
      CALL_RESHAPE_AND_CACHE_AGG(float, float, false);
    } else if (key.dtype() == at::ScalarType::Half) {
      CALL_RESHAPE_AND_CACHE_AGG(uint16_t, uint16_t, false);
    } else if (key.dtype() == at::ScalarType::BFloat16) {
      CALL_RESHAPE_AND_CACHE_AGG(__nv_bfloat16, __nv_bfloat16, false);
    }
  } else if (kv_cache_dtype == "fp8_e5m2") {
    if (key.dtype() == at::ScalarType::Float) {
      CALL_RESHAPE_AND_CACHE_AGG(float, uint8_t, true);
    } else if (key.dtype() == at::ScalarType::Half) {
      CALL_RESHAPE_AND_CACHE_AGG(uint16_t, uint8_t, true);
    } else if (key.dtype() == at::ScalarType::BFloat16) {
      CALL_RESHAPE_AND_CACHE_AGG(__nv_bfloat16, uint8_t, true);
    }
  } else {
    TORCH_CHECK(false, "Unsupported data type of kv cache: ", kv_cache_dtype);
  }
}

namespace vllm {

template<typename Tout, typename Tin>
__global__ void convert_fp8_e5m2_agg_kernel(
  const Tin* __restrict__ src_cache,
  Tout* __restrict__ dst_cache,
  const int64_t block_stride) {
  const int64_t block_idx = blockIdx.x;
  for (int i = threadIdx.x; i < block_stride; i += blockDim.x) {
    int64_t idx = block_idx * block_stride + i;
#ifdef ENABLE_FP8_E5M2
    dst_cache[idx] = fp8_e5m2_unscaled::vec_conversion<Tout, Tin>(src_cache[idx]);
#else
    assert(false);
#endif
  }
}

} // namespace vllm

#define CALL_CONVERT_FP8_E5M2_AGG(Tout, Tin)                                 \
  vllm::convert_fp8_e5m2_agg_kernel<Tout, Tin><<<grid, block, 0, stream>>>(  \
    reinterpret_cast<Tin*>(src_cache.data_ptr()),                        \
    reinterpret_cast<Tout*>(dst_cache.data_ptr()),                       \
    block_stride);

void convert_fp8_e5m2_agg(
  torch::Tensor& src_cache,
  torch::Tensor& dst_cache)
{
  int64_t num_blocks = src_cache.size(0);
  int64_t block_stride = src_cache.stride(0);

  dim3 grid(num_blocks);
  dim3 block(std::min(block_stride, int64_t(512)));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (src_cache.dtype() == at::ScalarType::Float) {
    CALL_CONVERT_FP8_E5M2_AGG(uint8_t, float);
  } else if (src_cache.dtype() == at::ScalarType::Half) {
    CALL_CONVERT_FP8_E5M2_AGG(uint8_t, uint16_t);
  } else if (src_cache.dtype() == at::ScalarType::BFloat16) {
    CALL_CONVERT_FP8_E5M2_AGG(uint8_t, __nv_bfloat16);
  } else if (dst_cache.dtype() == at::ScalarType::Float) {
    CALL_CONVERT_FP8_E5M2_AGG(float, uint8_t);
  } else if (dst_cache.dtype() == at::ScalarType::Half) {
    CALL_CONVERT_FP8_E5M2_AGG(uint16_t, uint8_t);
  } else if (dst_cache.dtype() == at::ScalarType::BFloat16) {
    CALL_CONVERT_FP8_E5M2_AGG(__nv_bfloat16, uint8_t);
  }
}