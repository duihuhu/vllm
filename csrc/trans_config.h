#ifndef TRANS_CONFIG_H
#define TRANS_CONFIG_H

#include <torch/torch.h>
#include <torch/cuda.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include "nccl.h"
#include "gpu_ops.h"
#include "trans_queue.h"

// 定义TaskType枚举类型，用于区分不同的任务类型
enum class TaskType {
    TRANSFER_SEND_BLOCKS,
    TRANSFER_RECV_BLOCKS,
};

class TransConfig {
public:
    int head_size;
    int num_heads;
    torch::Dtype dtype;
    int cache_size_per_block;

    TransConfig(int head_size, int num_heads, torch::Dtype dtype, int cache_size_per_block);
};

// TransferTaskMeta结构体，用于存储传输任务的元信息
class TransferTaskMeta {
public:
    TransferTaskMeta(const std::string& channel, const std::string& request_id)
        : channel(channel), request_id(request_id) {}
    std::string channel;
    std::string request_id;
};

class TransferTask {
public:
    TransferTask(const TransferTaskMeta& meta, const std::vector<int>& blocks, const std::vector<int>& opposite_ranks)
        : meta(meta), blocks(blocks), opposite_ranks(opposite_ranks) {}
    TransferTaskMeta meta;
    std::vector<int> blocks;
    std::vector<int> opposite_ranks;
};


// TransEngine类，负责管理KV缓存并执行发送和接收操作
class TransEngine {
public:
    TransEngine(const TransConfig& trans_config, const std::vector<torch::Tensor>& gpu_cache);

    void recv_blocks(const std::string& channel, const std::string& request_id,
                     const std::vector<int>& src_blocks, int opposite_rank);
    void send_blocks(const std::string& channel, const std::string& request_id,
                     const std::vector<int>& dst_blocks, int opposite_rank);
    std::vector<TransferTaskMeta> check_send_finished_events();
    std::vector<TransferTaskMeta> check_recv_finished_events();
private:
    TransConfig trans_config; // Add this member variable
    std::vector<torch::Tensor> gpu_cache; // Add this member variable

    int cache_size_per_block;

    std::unordered_map<std::string, c10::cuda::CUDAStream*> send_streams;
    std::unordered_map<std::string, std::vector<std::pair<std::string, at::cuda::CUDAEvent*>>> send_events;

    std::unordered_map<std::string, c10::cuda::CUDAStream*> recv_streams;
    std::unordered_map<std::string, std::vector<std::pair<std::string, at::cuda::CUDAEvent*>>> recv_events;
};

class TransWorker {
public:
    TransWorker(const TransConfig& trans_config, const std::vector<torch::Tensor>& gpu_cache,
                int rank, int local_rank, int nccl_local_rank);
    ~TransWorker();

    void add_tasks(const std::vector<std::pair<TaskType, TransferTask>>& tasks, bool is_prior);
    std::vector<std::pair<std::vector<TransferTaskMeta>, std::vector<TransferTaskMeta>>> get_finished_transfer_tasks();

private:
    void init_device();
    void worker();

    TransEngine trans_engine;
    TransQueue<std::pair<TaskType, TransferTask>> task_queue;
    TransQueue<std::pair<std::vector<TransferTaskMeta>, std::vector<TransferTaskMeta>>> transfer_result_queue;

    std::thread execute;
    int rank;
    int local_rank;
    int nccl_local_rank;
    ncclComm_t comm;
};
#endif // TRANS_CONFIG_H
