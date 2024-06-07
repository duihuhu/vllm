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
    while (true) {
        if(!worker_task_queue.empty()) {
            auto worker_task = worker_task_queue.pop_front();
            TransWorker* task_worker = trans_workers[worker_task.meta.channel];
            task_worker->add_tasks(worker_task);
        }
    }

    return;
}

std::vector<char> TransManager::get_nccl_id(const std::string& dst_channel){
    ncclUniqueId uniqueId; 
    ncclGetUniqueId(&uniqueId);
    if(trans_workers.find(dst_channel) == trans_workers.end()){
        TransWorker* task_worker = new TransWorker(cache_size_per_block, gpu_cache, rank, local_rank, nccl_local_rank, dst_channel, tp);
        trans_workers[dst_channel] = task_worker;
    }
    return std::vector<char>(uniqueId.internal, uniqueId.internal + sizeof(uniqueId.internal));
}

void TransManager::add_tasks(const std::vector<std::string>& tasks) {
    for (const auto& task : tasks) {
        auto trans_task = TransferTask::deserialize(task);
        worker_task_queue.push_back(trans_task);
    }
}
void TransManager::create_comm(std::vector<char>& nccl_id ,const std::string& dst_channel){
    if(trans_workers.find(dst_channel) == trans_workers.end()){
        TransWorker* task_worker = new TransWorker(cache_size_per_block, gpu_cache, rank, local_rank, nccl_local_rank, dst_channel, tp);
        trans_workers[dst_channel] = task_worker;
    }
    TransWorker* task_worker = trans_workers[dst_channel];
    task_worker->add_comm_task(nccl_id);
    return;
}

std::vector<std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>>> TransManager::get_finished_transfer_tasks() {
    std::vector<std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>>> finished_work_tasks;
    for (const auto& pair : trans_workers) {
        const std::string& key = pair.first;
        TransWorker* worker = pair.second;
        auto finished_work_task = worker->get_finished_transfer_tasks();
        if(!finished_work_task.empty()) {
            finished_work_tasks.emplace_back(finished_work_task);
        }
    }
    return finished_work_tasks;
}