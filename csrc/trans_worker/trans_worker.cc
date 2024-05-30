#include "trans_config.h"
#include "gpu_ops.h"
TransWorker::TransWorker(const TransConfig& trans_config, const std::vector<torch::Tensor>& gpu_cache,
                         int rank, int local_rank, int nccl_local_rank)
    : trans_engine(trans_config, gpu_cache),
      rank(rank), local_rank(local_rank), nccl_local_rank(nccl_local_rank) {
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
    if (CreateGlobalNcclComm(nccl_local_rank, 2, 0)!=0) {
        throw std::runtime_error("CreateNcclFromRankTable error");
    }
    
    while (true) {
        if(!task_queue.empty()) {
            auto task_pair = task_queue.pop_front();

            TaskType task_type = task_pair.first;
            TransferTask task = task_pair.second;
            auto task_meta = task.meta;
            switch (task_type) {
                case TaskType::TRANSFER_SEND_BLOCKS:
                    trans_engine.send_blocks(task_meta.channel, task_meta.request_id, task.blocks, task.opposite_ranks[rank]);
                    break;
                case TaskType::TRANSFER_RECV_BLOCKS:
                    trans_engine.recv_blocks(task_meta.channel, task_meta.request_id, task.blocks, task.opposite_ranks[rank]);
                    break;
                default:
                    throw std::runtime_error("invalid task_type.");
            }
        }
        auto send_blocks_finished = trans_engine.check_send_finished_events();
        auto recv_blocks_finished = trans_engine.check_recv_finished_events();
        transfer_result_queue.push(std::make_pair(send_blocks_finished, recv_blocks_finished));
    }
}

void TransWorker::add_tasks(const std::vector<std::pair<TaskType, TransferTask>>& tasks, bool is_prior) {
    for (const auto& task : tasks) {
        if (is_prior)
            task_queue.push_front(task);
        else
            task_queue.push_back(task)
    }
}

std::vector<std::pair<std::vector<TransferTaskMeta>, std::vector<TransferTaskMeta>>> TransWorker::get_finished_transfer_tasks() {
    std::vector<std::pair<std::vector<TransferTaskMeta>, std::vector<TransferTaskMeta>>> finished_tasks;
    while (!transfer_result_queue.empty())
    {
        auto finished_task = transfer_result_queue.pop_front();
        finished_tasks.push_back(finished_task);
    }
    
    return transfer_result_queue;
}