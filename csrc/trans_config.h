#ifndef TRANS_CONFIG_H
#define TRANS_CONFIG_H

#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/cuda.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include "nccl.h"
#include <nlohmann/json.hpp>  // Include the JSON library
#include <iostream>
#include <cuda_runtime.h>
#include "swap_config.h"
#include <tuple>

using json = nlohmann::json;
#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

// 定义TaskType枚举类型，用于区分不同的任务类型
enum class TaskType {
    TRANSFER_SEND_BLOCKS,
    TRANSFER_RECV_BLOCKS,
    TRANSFER_SEND_LAYER_BLOCKS,
    TRANSFER_RECV_LAYER_BLOCKS,
    //use in full blocks 
    TRANSFER_SEND_FULL_BLOCKS,
    TRANSFER_RECV_FULL_BLOCKS,

    //trans hbm to dram
    TRANSFER_HBM_TO_DRAM_BLOCKS,
    TRANSFER_HBM_TO_DRAM_FULL_BLOCKS,
};

// TransferTaskMeta结构体，用于存储传输任务的元信息
class TransferTaskMeta {
public:
    TransferTaskMeta(){}

    TransferTaskMeta(const std::string& channel, const std::string& request_id)
        : channel(channel), request_id(request_id) {}
    std::string channel;
    std::string request_id;

    // Serialize TransferTaskMeta to JSON
    json to_json() const {
        return json{{"channel", channel}, {"request_id", request_id}};
    }

    // Serialize the TransferTask to a string (JSON format)
    std::string serialize() const {
        json task_meta;
        task_meta["channel"] = channel;
        task_meta["request_id"] = request_id;
        return task_meta.dump();
    }

    static TransferTaskMeta deserialize(const std::string& serialized_data) {
        json task_meta = json::parse(serialized_data); // Parse JSON string
        return TransferTaskMeta(task_meta.at("channel").get<std::string>(), task_meta.at("request_id").get<std::string>());
    }
    // Deserialize TransferTaskMeta from JSON
    static TransferTaskMeta from_json(const json& task_meta) {
        return TransferTaskMeta{task_meta.at("channel").get<std::string>(), task_meta.at("request_id").get<std::string>()};
    }
};

class TransferTask {
public:
    TransferTask(const TransferTaskMeta& meta, const std::vector<uint32_t>& blocks, TaskType type, int layer = 1, bool is_last_layer=false)
        : meta(meta), blocks(blocks), type(type), layer(layer), is_last_layer(is_last_layer) {}

    TransferTask(const TransferTaskMeta& meta, const std::vector<uint32_t>& blocks, const std::vector<uint32_t>& dst_blocks, TaskType type, int layer = 1, bool is_last_layer=false)
        : meta(meta), blocks(blocks), dst_blocks(dst_blocks), type(type), layer(layer), is_last_layer(is_last_layer) {}
    
    TransferTaskMeta meta;
    std::vector<uint32_t> blocks;
    std::vector<uint32_t> dst_blocks;
    TaskType type;
    int layer;
    bool is_last_layer;

    // Serialize the TransferTask to a string (JSON format)
    std::string serialize() const {
        json task;
        task["meta"] = meta.to_json();
        task["blocks"] = blocks;
        if (!dst_blocks.empty()) {
            task["dst_blocks"] = dst_blocks;
        }
        task["type"] = static_cast<int>(type);  // Store TaskType as an integer
        task["layer"] = layer;  // Store TaskType as an integer
        task["is_last_layer"] = is_last_layer;  // Store TaskType as an integer
        return task.dump();
    }

    // Deserialize from a string (JSON format) to a TransferTask object
    static TransferTask deserialize(const std::string& serialized_data) {
        json task = json::parse(serialized_data);
        TransferTaskMeta meta = TransferTaskMeta::from_json(task.at("meta"));
        std::vector<uint32_t> blocks = task.at("blocks").get<std::vector<uint32_t>>();
        std::vector<uint32_t> dst_blocks;
        if (task.contains("dst_blocks")) {
            dst_blocks = task.at("dst_blocks").get<std::vector<uint32_t>>();
        }
        TaskType type = static_cast<TaskType>(task.at("type").get<int>());
        int layer = task.at("layer").get<int>();
        bool is_last_layer = task.at("is_last_layer").get<bool>();

        if (dst_blocks.empty()) {
            return TransferTask(meta, blocks, type, layer, is_last_layer);
        } else {
            return TransferTask(meta, blocks, dst_blocks, type, layer, is_last_layer);
        }
    }
};


// TransEngine类，负责管理KV缓存并执行发送和接收操作
class TransEngine {
public:
    TransEngine(int cache_size_per_block, const std::vector<std::pair<at::Tensor, at::Tensor>>& gpu_cache, int cache_block_size, std::vector<uint64_t>& blocks_gpu_cache);

    TransEngine(int cache_size_per_block, const std::vector<std::pair<at::Tensor, at::Tensor>>& gpu_cache, int cache_block_size, std::vector<uint64_t>& blocks_gpu_cache, const std::vector<std::pair<at::Tensor, at::Tensor>>& dst_cpu_cache);

    void recv_blocks(const std::string& channel, const std::string& request_id, const std::vector<uint32_t>& src_blocks, int opposite_rank, ncclComm_t& comm, c10::cuda::CUDAStream& stream);
    
    void send_blocks(const std::string& channel, const std::string& request_id,const std::vector<uint32_t>& dst_blocks, int opposite_rank, ncclComm_t& comm, c10::cuda::CUDAStream& stream);
    
    void send_layer_blocks(const std::string& channel, const std::string& request_id, const std::vector<uint32_t>& dst_blocks, int opposite_rank, int layer, bool is_last_layer, ncclComm_t& comm, c10::cuda::CUDAStream& stream);
    
    void recv_layer_blocks(const std::string& channel, const std::string& request_id, const std::vector<uint32_t>& src_blocks, int opposite_rank, int layer , bool is_last_layer ,ncclComm_t& comm, c10::cuda::CUDAStream& stream);
    
    void send_comms_layer_blocks(const std::string& channel, const std::string& request_id, const std::vector<uint32_t>& dst_blocks, int opposite_rank, int layer,  bool is_last_layer, ncclComm_t& comm, c10::cuda::CUDAStream& stream, int comm_id);

    void recv_comms_layer_blocks(const std::string& channel, const std::string& request_id, const std::vector<uint32_t>& src_blocks, int opposite_rank, int layer, bool is_last_layer, ncclComm_t& comm, c10::cuda::CUDAStream& stream, int comm_id);

    void send_full_blocks(const std::string& channel, const std::string& request_id, const std::vector<uint32_t>& dst_blocks, int opposite_rank, ncclComm_t& comm, c10::cuda::CUDAStream& stream);

    void recv_full_blocks(const std::string& channel, const std::string& request_id, const std::vector<uint32_t>& src_blocks, int opposite_rank, ncclComm_t& comm, c10::cuda::CUDAStream& stream);

    int create_nccl_comm(int32_t rank, ncclComm_t& comm, ncclUniqueId& uniqueId , int32_t NumDevice);

    void swap_hbm_to_remote_dram_blocks(const std::string& channel, const std::string& request_id, const std::vector<uint32_t>& blocks, std::vector<uint32_t>& dst_blocks, c10::cuda::CUDAStream& stream);


    std::vector<std::string> check_send_finished_events();
    std::vector<std::string> check_recv_finished_events();

    std::vector<std::string> check_send_finished_comms_events();
    std::vector<std::string> check_recv_finished_comms_events();
    std::vector<std::string> check_swap_remote_finished_events();

    void SendBlocks(std::vector<std::pair<at::Tensor, at::Tensor>>& srcCaches, \
        const std::vector<uint32_t>& srcBlocks, uint32_t cacheSize, uint32_t destRank, ncclComm_t& comm);
    void RecvBlocks(std::vector<std::pair<at::Tensor, at::Tensor>>& dstCaches, \
        const std::vector<uint32_t>& dstBlocks, uint32_t cacheSize, uint32_t srcRank, ncclComm_t& comm);

    void SendLayerBlocks(std::vector<std::pair<at::Tensor, at::Tensor>>& srcCaches, \
        const std::vector<uint32_t>& srcBlocks, uint32_t cacheSize, uint32_t destRank, uint32_t layer, ncclComm_t& comm);
    void RecvLayerBlocks(std::vector<std::pair<at::Tensor, at::Tensor>>& dstCaches, \
    const std::vector<uint32_t>& dstBlocks, uint32_t cacheSize, uint32_t srcRank, uint32_t layer, ncclComm_t& comm);

    void RecvFullBlocks(std::vector<uint64_t>& dstCaches, \
        const std::vector<uint32_t>& dstBlocks, uint32_t cacheSize, uint32_t srcRank, ncclComm_t& comm);
    void SendFullBlocks(std::vector<uint64_t>& srcCaches, \
        const std::vector<uint32_t>& srcBlocks, uint32_t cacheSize, uint32_t destRank, ncclComm_t& comm);

    void SwapHbmToRemoteDramBlocks(std::vector<std::pair<at::Tensor, at::Tensor>>& srcCaches, \
        std::vector<std::pair<at::Tensor, at::Tensor>>& dstCaches, const std::vector<uint32_t>& srcBlocks, const std::vector<uint32_t>& dstBlocks, uint32_t cacheSize);
private:
    std::vector<std::pair<at::Tensor, at::Tensor>> gpu_cache;
    std::vector<uint64_t> blocks_gpu_cache; // key/value address in tensor 

    std::vector<std::pair<at::Tensor, at::Tensor>> dst_cpu_cache;

    int cache_size_per_block;
    int cache_block_size;
    // std::unordered_map<std::string, c10::cuda::CUDAStream*> send_streams;
    std::unordered_map<std::string, std::vector<std::pair<std::string, at::cuda::CUDAEvent*>>> send_events;
    // std::unordered_map<std::string, c10::cuda::CUDAStream*> recv_streams;
    std::unordered_map<std::string, std::vector<std::pair<std::string, at::cuda::CUDAEvent*>>> recv_events;

    std::unordered_map<std::string, std::unordered_map<int, std::vector<at::cuda::CUDAEvent*>>> send_req_events;
    std::unordered_map<std::string, std::unordered_map<int, std::vector<at::cuda::CUDAEvent*>>> recv_req_events;


    std::unordered_map<std::string, std::vector<std::pair<std::string, std::unordered_map<int, std::vector<at::cuda::CUDAEvent*>>>>> send_comms_events;
    std::unordered_map<std::string, std::vector<std::pair<std::string, std::unordered_map<int, std::vector<at::cuda::CUDAEvent*>>>>> recv_comms_events;

    //swao to remote instance dram
    std::unordered_map<std::string, std::vector<std::pair<std::string, at::cuda::CUDAEvent*>>> swap_remote_events;

};

class TransWorker {
public:

    TransWorker(int cache_size_per_block, const std::vector<std::pair<at::Tensor, at::Tensor>>& gpu_cache, int rank, int local_rank, int nccl_local_rank, const std::string& dst_channel, int tp, int num_layer, int cache_block_size, std::vector<uint64_t>& blocks_gpu_cache);

    TransWorker(int cache_size_per_block, const std::vector<std::pair<at::Tensor, at::Tensor>>& gpu_cache, int rank, int local_rank, int nccl_local_rank, const std::string& dst_channel, int tp, int num_layer, int cache_block_size, std::vector<uint64_t>& blocks_gpu_cache, const std::vector<std::pair<at::Tensor, at::Tensor>>& dst_cpu_cache);


    ~TransWorker();

    void add_tasks(TransferTask& task);
    // void add_tasks(const std::vector<std::string>& tasks);
    // std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> get_finished_transfer_tasks();
    std::vector<std::tuple<std::vector<std::string>, std::vector<std::string>,std::vector<std::string>>> get_finished_transfer_tasks();

    void add_comm_task(std::vector<char>& uniqueId);
private:
    void init_device();
    void worker();

    TransEngine trans_engine;
    TransQueue<TransferTask> task_queue;
    TransQueue<std::vector<char>> comm_queue;
    TransQueue<std::tuple<std::vector<std::string>, std::vector<std::string>,std::vector<std::string>>> transfer_result_queue;

    std::thread execute;
    int rank;
    int local_rank;
    int nccl_local_rank;
    std::string dst_channel;
    std::vector<int> dst_ranks;
    int comm_rank;
    int dst_rank;
    int tp;
    int num_layer;
    std::vector<ncclComm_t> comms;
    std::vector<c10::cuda::CUDAStream> streams;
    int use_comm;

    std::vector<c10::cuda::CUDAStream> swap_remote_streams;
    int use_swap_stream;
};

class TransManager {
public:
    TransManager(int cache_size_per_block, std::vector<std::pair<at::Tensor, at::Tensor>>& gpu_cache, int rank, int local_rank, int nccl_local_rank, int tp, int num_layer, int cache_block_size, std::vector<uint64_t>& blocks_gpu_cache);


    ~TransManager();
    std::vector<char> get_nccl_id(const std::string& dst_channel, const std::string& worker_type);
    void create_comm(std::vector<char>& nccl_id ,const std::string& dst_channel, const std::string& worker_type);
    void add_tasks(const std::vector<std::string>& tasks);
    void dist_worker();
    std::vector<std::vector<std::tuple<std::vector<std::string>, std::vector<std::string>,std::vector<std::string>>>> get_finished_transfer_tasks();

    void init_dst_cpu_cache(const std::string& dst_channel, const std::vector<std::pair<at::Tensor, at::Tensor>>& dst_cpu_cache);

private:
    std::unordered_map<std::string, TransWorker*> send_trans_workers;

    std::unordered_map<std::string, TransWorker*> recv_trans_workers;

    std::unordered_map<std::string, TransWorker*> swap_workers;

    std::thread execute;

    int cache_size_per_block;
    std::vector<std::pair<at::Tensor, at::Tensor>> gpu_cache;
    std::vector<uint64_t> blocks_gpu_cache;

    TransQueue<TransferTask> worker_task_queue;
    int rank;
    int local_rank;
    int nccl_local_rank;
    int tp;
    int num_layer;
    int cache_block_size;

};
#endif // TRANS_CONFIG_H
