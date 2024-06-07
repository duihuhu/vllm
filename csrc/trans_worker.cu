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
    if (CreateGlobalMulNcclComm(nccl_local_rank, 4, 4)!=0) {
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
                case TaskType::TRANSFER_SEND_LAYER_BLOCKS:
                    // std::cout<< "send_layer_blocks " << task.layer << " " << task.is_last_layer;
                    trans_engine.send_layer_blocks(task_meta.channel, task_meta.request_id, task.blocks, task.opposite_ranks[rank], task.layer, task.is_last_layer);
                    break;
                case TaskType::TRANSFER_RECV_LAYER_BLOCKS:
                    trans_engine.recv_layer_blocks(task_meta.channel, task_meta.request_id, task.blocks, task.opposite_ranks[rank], 40);
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

// void TransWorker::add_tasks(const std::vector<TransferTask>& tasks) {
//     for (const auto& task : tasks) {
//         task_queue.push_back(task);
//     }
// }

void TransWorker::add_tasks(const std::vector<std::string>& tasks) {
    for (const auto& task : tasks) {
        auto trans_task = TransferTask::deserialize(task);
        task_queue.push_back(trans_task);
    }
}

std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> TransWorker::get_finished_transfer_tasks() {
    std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> finished_tasks;
    while (!transfer_result_queue.empty())
    {
        // std::cout<<"transfer_result_queue is not empty ";
        auto finished_task = transfer_result_queue.pop_front();
        finished_tasks.push_back(finished_task);
    }
    return finished_tasks;
}

std::vector<char> TransWorker::get_nccl_id(std::string dst_channel){
    ncclUniqueId uniqueId; 
    // int shmSize = sizeof(ncclUniqueId);
    ncclGetUniqueId(&uniqueId);
    comm_queue.push_back(CommTask(uniqueId, dst_channel));
    std::cout << "NCCL Unique ID get in C++: " << " nccl_local_rank " << nccl_local_rank << std::endl;
    for (char c : uniqueId.internal) {
        std::cout << std::hex << (int)c << " ";
    }
    return std::vector<char>(uniqueId.internal, uniqueId.internal + sizeof(uniqueId.internal));
}

bool TransWorker::create_comm(std::vector<char> nccl_id, std::string dst_channel){
    ncclUniqueId uniqueId;
    std::memcpy(uniqueId.internal, nccl_id.data(), sizeof(uniqueId.internal));
    ncclComm_t comm;
    std::cout << "NCCL Unique ID set in C++: " << " nccl_local_rank " << nccl_local_rank << std::endl;
    for (char c : uniqueId.internal) {
        std::cout << std::hex << (int)c << " ";
    }
    // if (CreateInternalNcclComm(nccl_local_rank, 4, comm, uniqueId)!=0) {
    //     throw std::runtime_error("CreateNcclFromRankTable error");
    // }
    return true;
}