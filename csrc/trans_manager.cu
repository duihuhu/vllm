#include "trans_config.h"
TransManager::TransManager(int cache_size_per_block, const std::vector<std::pair<at::Tensor, at::Tensor>>& gpu_cache, int rank, int local_rank, int nccl_local_rank)
    : cache_size_per_block(cache_size_per_block), gpu_cache(gpu_cache), rank(rank), local_rank(local_rank), nccl_local_rank(nccl_local_rank) {
    std::cout << "trans manager " << " rank " << rank << " local_rank " << local_rank << " nccl_local_rank " << nccl_local_rank<<std::endl;
    execute = std::thread(&TransManager::dist_worker, this);
}

TransManager::~TransManager() {
    if (execute.joinable()) {
        execute.join();
    }
}
void TransManager::dist_worker() {
}

std::vector<char> TransManager::get_nccl_id(std::string dst_channel){
    ncclUniqueId uniqueId; 
    ncclGetUniqueId(&uniqueId);
    std::cout << "NCCL Unique ID get in C++: " << " nccl_local_rank " << nccl_local_rank << std::endl;
    for (char c : uniqueId.internal) {
        std::cout << std::hex << (int)c << " ";
    }
    if(trans_workers.find(dst_channel) == trans_workers.end()){
        trans_workers.emplace(dst_channel, TransWorker(cache_size_per_block, gpu_cache, rank, local_rank, nccl_local_rank););
    }else{
         TransWorker& trans_worker = trans_workers[dst_channel];
        // trans_worker.add_tasks();
    }
    return std::vector<char>(uniqueId.internal, uniqueId.internal + sizeof(uniqueId.internal));
}
