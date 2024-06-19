#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "attention_dtypes.h"
#include "attention_utils.cuh"
#ifdef ENABLE_FP8_E5M2
#include "../quantization/fp8_e5m2_kvcache/quant_utils.cuh"
#endif

#include <algorithm>
#include <vector>

torch::Tensor tensor_for_caches_addresses(
    std::vector<torch::Tensor>& caches
) {
    int num_blocks = caches.size();
    TORCH_CHECK(num_blocks > 0, "Cache Blocks are Empty!")

    torch::Device caches_device = caches[0].device();
    //TORCH_CHECK(caches_device.is_cuda());
    //The swap_block_agg will use a tensor in CPU for the addresses of kv caches in CPU

    //int64_t caches_ptrs[num_blocks];
    std::vector<int64_t> caches_ptrs(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
        caches_ptrs[i] = reinterpret_cast<int64_t>(caches[i].data_ptr());
        // std::cout<<"tensor_for_caches_addresses " << caches_ptrs[i]<<std::endl;
    }

    torch::Tensor caches_ptrs_tensor = torch::from_blob(
        // caches_ptrs.data(), {num_blocks}, torch::kInt64).to(caches_device).clone();
        caches_ptrs.data(), {num_blocks}, torch::kInt64).to(caches_device);
    return caches_ptrs_tensor;
}


std::vector<uint64_t> tensor_for_blocks_address(
    std::vector<torch::Tensor>& caches
) {
    int num_blocks = caches.size();
    TORCH_CHECK(num_blocks > 0, "Cache Blocks are Empty!")

    torch::Device caches_device = caches[0].device();
    std::vector<uint64_t> caches_ptrs(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
        caches_ptrs[i] = reinterpret_cast<uint64_t>(caches[i].data_ptr());
        // std::cout << " tensor_for_caches_addresses " << caches_ptrs[i] <<std::endl;
    }
    return caches_ptrs;
}
