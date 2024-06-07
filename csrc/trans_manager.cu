#include "trans_config.h"
TransManager::TransManager(int cache_size_per_block, std::vector<std::pair<at::Tensor, at::Tensor>>& gpu_cache, int rank, int local_rank, int nccl_local_rank, int tp): cache_size_per_block(cache_size_per_block), gpu_cache(gpu_cache), rank(rank), local_rank(local_rank), nccl_local_rank(nccl_local_rank), tp(tp) {
    execute = std::thread(&TransManager::dist_worker, this);
}

TransManager::~TransManager() {

    if (execute.joinable()) {
        execute.join();
    }
}
void TransManager::dist_worker() {
    return;
}

std::vector<char> TransManager::get_nccl_id(const std::string& dst_channel){
    ncclUniqueId uniqueId; 
    ncclGetUniqueId(&uniqueId);
    for (char c : uniqueId.internal) {
        std::cout << std::hex << (int)c << " ";
    }
    if(trans_workers.find(dst_channel) == trans_workers.end()){
        TransWorker* task_worker = new TransWorker(cache_size_per_block, gpu_cache, rank, local_rank, nccl_local_rank, dst_channel, tp);
        trans_workers[dst_channel] = task_worker;
    }
    return std::vector<char>(uniqueId.internal, uniqueId.internal + sizeof(uniqueId.internal));
}

void TransManager::create_comm(std::vector<char>& nccl_id ,const std::string& dst_channel){
    ncclUniqueId uniqueId;
    std::memcpy(uniqueId.internal, nccl_id.data(), sizeof(uniqueId.internal));
    std::cout << "NCCL Unique ID set in C++: " << " nccl_local_rank " << nccl_local_rank << std::endl;
    for (char c : uniqueId.internal) {
        std::cout << std::hex << (int)c << " ";
    }
    TransWorker* task_worker = trans_workers[dst_channel];
    task_worker->add_comm_task(uniqueId);
    // if (CreateInternalNcclComm(nccl_local_rank, 4, comm, uniqueId)!=0) {
    //     throw std::runtime_error("CreateNcclFromRankTable error");
    // }
    return;
}
