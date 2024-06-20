#include "swap_config.h"

SwapManager::SwapManager(int cache_size_per_block, std::vector<std::pair<at::Tensor, at::Tensor>> &gpu_cache, std::vector<std::pair<at::Tensor, at::Tensor>> &cpu_cache, bool is_layer, int layerNum): cache_size_per_block(cache_size_per_block), gpu_cache(gpu_cache), cpu_cache(cpu_cache), is_layer(is_layer), layerNum(layerNum) {
    swap_out_streams.push_back(c10::cuda::CUDAStream(c10::cuda::getStreamFromPool(true)));

    execute = std::thread(&SwapManager::worker, this);
}

SwapManager::SwapManager(int cache_block_size, std::vector<uint64_t>& gpu_blocks_address, std::vector<uint64_t>& cpu_blocks_address): cache_block_size(cache_block_size), gpu_blocks_address(gpu_blocks_address), cpu_blocks_address(cpu_blocks_address){
    swap_out_streams.push_back(c10::cuda::CUDAStream(c10::cuda::getStreamFromPool(true)));

    execute = std::thread(&SwapManager::worker, this);
}

SwapManager::~SwapManager()
{
    if (execute.joinable()) {
        execute.join();
    }
}

void SwapManager::worker() {
    while (true) {
        if(!task_queue.empty()) {
            auto task = task_queue.pop_front();
            auto swap_id = task.swap_id;
            auto evicted_blocks = task.evicted_blocks;
            SwapType swap_type = task.type;
            switch (swap_type) {
                case SwapType::SWAP_OUT_BLOCKS:
                    swap_out(swap_id, evicted_blocks);
                    break;
                case SwapType::SWAP_IN_BLOCKS:
                    break;
                case SwapType::SWAP_OUT_FULL_BLOCKS:
                    swap_out_full_blocks(swap_id, evicted_blocks);
                    break;
                default:
                    throw std::runtime_error("invalid swap_type.");
            }
        }
        auto swap_blocks_finished = check_finished_swap_out_events();
        if (!swap_blocks_finished.empty()){
            swap_result_queue.push_back(swap_blocks_finished);
        }
    }
}

void SwapManager::add_swap_tasks(const SwapTask& task) {
    task_queue.push_back(task);
}

void SwapManager::swap_out(const std::string& swap_id, const std::map<int, int>& evicted_blocks){
    c10::cuda::CUDAStreamGuard guard(swap_out_streams[0]);
    cudaMemcpyKind memcpy_type = cudaMemcpyDeviceToHost;
    for (int i=0; i < layerNum; i++) {
        at::Tensor srcKeyCache = gpu_cache[i].first;
        at::Tensor srcValueCache = gpu_cache[i].second;

        at::Tensor dstKeyCache = cpu_cache[i].first;
        at::Tensor dstValueCache = cpu_cache[i].second;
        for (const auto& pair : evicted_blocks) {
            void *srcKeyCachePtr = srcKeyCache.index({pair.first}).data_ptr();
            void *dstKeyCachePtr = dstKeyCache.index({pair.second}).data_ptr();
            cudaMemcpyAsync(
            srcKeyCachePtr,
            dstKeyCachePtr,
            cache_size_per_block,
            memcpy_type,
            swap_out_streams[0]);

            void *srcValueCachePtr = srcValueCache.index({pair.first}).data_ptr();
            void *dstValueCachePtr = dstValueCache.index({pair.second}).data_ptr();

            cudaMemcpyAsync(
            srcValueCachePtr,
            dstValueCachePtr,
            cache_size_per_block,
            memcpy_type,
            swap_out_streams[0]);
        }
    }
    at::cuda::CUDAEvent* event = new at::cuda::CUDAEvent();
    event->record();
    swap_out_events[swap_id] = event;
}

void SwapManager::swap_out_full_blocks(const std::string& swap_id, const std::map<int, int>& evicted_blocks){
    c10::cuda::CUDAStreamGuard guard(swap_out_streams[0]);
    cudaMemcpyKind memcpy_type = cudaMemcpyDeviceToHost;
    for (const auto& pair : evicted_blocks) {
        void *srcCachePtr = (void*)gpu_blocks_address[pair.first];
        void *dstCachePtr = (void*)cpu_blocks_address[pair.first];

        cudaMemcpyAsync(
        srcCachePtr,
        dstCachePtr,
        cache_block_size,
        memcpy_type,
        swap_out_streams[0]);
    }
    at::cuda::CUDAEvent* event = new at::cuda::CUDAEvent();
    event->record();
    swap_out_events[swap_id] = event;
}


std::vector<std::string> SwapManager::check_finished_swap_out_events(){
    std::vector<std::string> swap_out_finished;

    auto it = swap_out_events.begin();
    while (it != swap_out_events.end()) {
        const std::string& swap_id = it->first;
        auto& event = it->second;
        if (event->query()) {
            swap_out_finished.emplace_back(swap_id);
            it = swap_out_events.erase(it);
        } else {
            ++it;
        }
    }
    return swap_out_finished;
}

std::vector<std::vector<std::string>> SwapManager::get_finished_swap_tasks() {
    std::vector<std::vector<std::string>> finished_tasks;
    while (!swap_result_queue.empty())
    {
        // std::cout<<"transfer_result_queue is not empty ";
        auto finished_task = swap_result_queue.pop_front();
        finished_tasks.push_back(finished_task);
    }
    return finished_tasks;
}