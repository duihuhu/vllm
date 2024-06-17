#include "trans_config.h"

TransManager::TransManager(int cache_size_per_block, std::vector<std::pair<at::Tensor, at::Tensor>>& gpu_cache, int rank, int local_rank, int nccl_local_rank, int tp, int num_layer, int cache_block_size, std::vector<uint64_t>& blocks_gpu_cache): cache_size_per_block(cache_size_per_block), gpu_cache(gpu_cache), rank(rank), local_rank(local_rank), nccl_local_rank(nccl_local_rank), tp(tp), num_layer(num_layer), cache_block_size(cache_block_size), blocks_gpu_cache(blocks_gpu_cache){
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
            TaskType task_type = worker_task.type;
            TransWorker* task_worker = nullptr;
            switch (task_type) {
                case TaskType::TRANSFER_SEND_BLOCKS:
                    task_worker = send_trans_workers[worker_task.meta.channel];
                    task_worker->add_tasks(worker_task);
                    break;
                case TaskType::TRANSFER_RECV_BLOCKS:
                    task_worker = recv_trans_workers[worker_task.meta.channel];
                    task_worker->add_tasks(worker_task);
                    break;
                case TaskType::TRANSFER_SEND_LAYER_BLOCKS:
                    task_worker = send_trans_workers[worker_task.meta.channel];
                    task_worker->add_tasks(worker_task);
                    break;
                case TaskType::TRANSFER_RECV_LAYER_BLOCKS:
                    task_worker = recv_trans_workers[worker_task.meta.channel];
                    task_worker->add_tasks(worker_task);
                    break;
                case TaskType::TRANSFER_SEND_FULL_BLOCKS:
                    task_worker = send_trans_workers[worker_task.meta.channel];
                    task_worker->add_tasks(worker_task);
                    break;
                case TaskType::TRANSFER_RECV_FULL_BLOCKS:
                    task_worker = recv_trans_workers[worker_task.meta.channel];
                    task_worker->add_tasks(worker_task);
                    break;
                case TaskType::TRANSFER_HBM_TO_DRAM_BLOCKS:
                    task_worker = swap_workers[worker_task.meta.channel];
                    task_worker->add_tasks(worker_task);
                    break;
                default:
                    throw std::runtime_error("invalid task_type.");
            }
        }
    }
    return;
}

std::vector<char> TransManager::get_nccl_id(const std::string& dst_channel, const std::string& worker_type){
    ncclUniqueId uniqueId; 
    ncclGetUniqueId(&uniqueId);
    if(worker_type=="sender"){
        if(send_trans_workers.find(dst_channel) == send_trans_workers.end()){
            TransWorker* task_worker = new TransWorker(cache_size_per_block, gpu_cache, rank, local_rank, nccl_local_rank, dst_channel, tp, num_layer, cache_block_size, blocks_gpu_cache);
            send_trans_workers[dst_channel] = task_worker;
        }
    } else{
        if(recv_trans_workers.find(dst_channel) == recv_trans_workers.end()){
            TransWorker* task_worker = new TransWorker(cache_size_per_block, gpu_cache, rank, local_rank, nccl_local_rank, dst_channel, tp, num_layer, cache_block_size, blocks_gpu_cache);
            recv_trans_workers[dst_channel] = task_worker;
        }
    }
    return std::vector<char>(uniqueId.internal, uniqueId.internal + sizeof(uniqueId.internal));
}

void TransManager::add_tasks(const std::vector<std::string>& tasks) {
    for (const auto& task : tasks) {
        auto trans_task = TransferTask::deserialize(task);
        worker_task_queue.push_back(trans_task);
    }
}
void TransManager::create_comm(std::vector<char>& nccl_id ,const std::string& dst_channel, const std::string& worker_type){
    if(worker_type=="sender"){
        if(send_trans_workers.find(dst_channel) == send_trans_workers.end()){
            TransWorker* task_worker = new TransWorker(cache_size_per_block, gpu_cache, rank, local_rank, nccl_local_rank, dst_channel, tp, num_layer, cache_block_size, blocks_gpu_cache);
            send_trans_workers[dst_channel] = task_worker;
        }
        TransWorker* task_worker = send_trans_workers[dst_channel];
        task_worker->add_comm_task(nccl_id);
    } else{
        if(recv_trans_workers.find(dst_channel) == recv_trans_workers.end()){
            TransWorker* task_worker = new TransWorker(cache_size_per_block, gpu_cache, rank, local_rank, nccl_local_rank, dst_channel, tp, num_layer, cache_block_size, blocks_gpu_cache);
            recv_trans_workers[dst_channel] = task_worker;
        }
        TransWorker* task_worker = recv_trans_workers[dst_channel];
        task_worker->add_comm_task(nccl_id);
    }
    return;
}

void TransManager::init_dst_cpu_cache(const std::string& dst_channel, const std::vector<std::pair<at::Tensor, at::Tensor>>& dst_cpu_cache) {
    if(swap_workers.find(dst_channel) == swap_workers.end()){
        std::cout << "dst_channel " << dst_channel << " " << nccl_local_rank << std::endl;
        TransWorker* task_worker = new TransWorker(cache_size_per_block, gpu_cache, rank, local_rank, nccl_local_rank, dst_channel, tp, num_layer, cache_block_size, blocks_gpu_cache, dst_cpu_cache);
        swap_workers[dst_channel] = task_worker;
    }
    return;
}

std::vector<std::vector<std::tuple<std::vector<std::string>, std::vector<std::string>,std::vector<std::string>>>>TransManager::get_finished_transfer_tasks() {
    std::vector<std::vector<std::tuple<std::vector<std::string>, std::vector<std::string>,std::vector<std::string>>>> finished_work_tasks;
    for (const auto& pair : send_trans_workers) {
        // const std::string& key = pair.first;
        TransWorker* worker = pair.second;
        auto finished_work_task = worker->get_finished_transfer_tasks();
        if(!finished_work_task.empty()) {
            finished_work_tasks.emplace_back(finished_work_task);
        }
    }
    for (const auto& pair : recv_trans_workers) {
        // const std::string& key = pair.first;
        TransWorker* worker = pair.second;
        auto finished_work_task = worker->get_finished_transfer_tasks();
        if(!finished_work_task.empty()) {
            finished_work_tasks.emplace_back(finished_work_task);
        }
    }

    for (const auto& pair : swap_workers) {
        // const std::string& key = pair.first;
        TransWorker* worker = pair.second;
        auto finished_work_task = worker->get_finished_transfer_tasks();
        if(!finished_work_task.empty()) {
            finished_work_tasks.emplace_back(finished_work_task);
        }
    }

    return finished_work_tasks;
}