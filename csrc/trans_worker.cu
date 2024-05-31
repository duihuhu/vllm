#include "trans_config.h"

TransWorker::TransWorker(int cache_size_per_block, const std::vector<std::pair<at::Tensor, at::Tensor>>& gpu_cache, int rank, int local_rank, int nccl_local_rank)
    : trans_engine(cache_size_per_block, gpu_cache), rank(rank), local_rank(local_rank), nccl_local_rank(nccl_local_rank) {
    execute = std::thread(&TransWorker::worker, this);
}

TransWorker::~TransWorker() {
    if (execute.joinable()) {
        execute.join();
    }
}

void TransWorker::init_device() {
    torch::Device device(torch::kCUDA, local_rank);
    c10::cuda::set_device(device.index());
}

void TransWorker::worker() {
    init_device();
    if (CreateGlobalNcclComm(nccl_local_rank, 4, 0)!=0) {
        throw std::runtime_error("CreateNcclFromRankTable error");
    }
    
    while (true) {
        if(!task_queue.empty()) {
            // std::cout<<"task_queue is not empty ";
            auto task = task_queue.pop_front();
            TaskType task_type = task.type;
            auto task_meta = task.meta;
            switch (task_type) {
                case TaskType::TRANSFER_SEND_BLOCKS:
                    // std::cout<<"task_queue is not empty send " <<std::endl;
                    trans_engine.send_blocks(task_meta.channel, task_meta.request_id, task.blocks, task.opposite_ranks[rank]);
                    break;
                case TaskType::TRANSFER_RECV_BLOCKS:
                    // std::cout<<"task_queue is not empty recv " <<std::endl;
                    trans_engine.recv_blocks(task_meta.channel, task_meta.request_id, task.blocks, task.opposite_ranks[rank]);
                    break;
                default:
                    throw std::runtime_error("invalid task_type.");
            }
        }
        // std::cout<<"task_queue is empty ";
        auto send_blocks_finished = trans_engine.check_send_finished_events();
        auto recv_blocks_finished = trans_engine.check_recv_finished_events();
        if (!send_blocks_finished.empty() || !recv_blocks_finished.empty()){
            // std::cout<<"task_queue is empty send " << send_blocks_finished.empty() << " recv " << recv_blocks_finished.empty()<<std::endl;
            transfer_result_queue.push_back(std::make_pair(send_blocks_finished, recv_blocks_finished));
        }
    }
}

void TransWorker::add_tasks(const std::vector<TransferTask>& tasks) {
    for (const auto& task : tasks) {
        task_queue.push_back(task);
    }
}

std::vector<std::pair<std::vector<TransferTaskMeta>, std::vector<TransferTaskMeta>>> TransWorker::get_finished_transfer_tasks() {
    std::vector<std::pair<std::vector<TransferTaskMeta>, std::vector<TransferTaskMeta>>> finished_tasks;
    while (!transfer_result_queue.empty())
    {
        // std::cout<<"transfer_result_queue is not empty ";
        auto finished_task = transfer_result_queue.pop_front();
        finished_tasks.push_back(finished_task);
    }
    return finished_tasks;
}