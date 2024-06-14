#ifndef SWAP_H
#define SWAP_H

#include "trans_config.h"

// 定义TaskType枚举类型，用于区分不同的任务类型
enum class SwapType {
    SWAP_OUT_BLOCKS,
    SWAP_IN_BLOCKS,
};

class SwapTask {
public:
    SwapTask(const std::string& swap_id, const std::map<int, int>& evicted_blocks, SwapType type): swap_id(swap_id), evicted_blocks(evicted_blocks), type(type){}
    std::string swap_id;
    std::map<int, int> evicted_blocks;
    SwapType type;
};

class SwapManager {
public:
    SwapManager(int cache_size_per_block, std::vector<std::pair<at::Tensor, at::Tensor>>& gpu_cache, std::vector<std::pair<at::Tensor, at::Tensor>>& cpu_cache, bool is_layer);
    ~SwapManager();
    void add_swap_tasks(const SwapTask& tasks);
    void worker();
    void swap_out(const std::string& swap_id, const std::map<int, int>& evicted_blocks);
    void check_finished_swap_out_events();
    std::vector<std::vector<std::string>> get_finished_swap_tasks();

private:
    std::vector<std::pair<at::Tensor, at::Tensor>> gpu_cache; // Add this member variable
    std::vector<std::pair<at::Tensor, at::Tensor>> cpu_cache; // Add this member variable
    int cache_size_per_block;
    bool is_layer;
    TransQueue<SwapTask> task_queue;
    TransQueue<std::vector<std::string>> swap_result_queue;
    std::unordered_map<std::string, at::cuda::CUDAEvent*> swap_out_events;
    int layerNum;
    c10::cuda::CUDAStream swap_out_stream;
    c10::cuda::CUDAStream swap_in_stream;
};


#endif // SWAP_H