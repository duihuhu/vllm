#include "trans_config.h"
#include <stdexcept>
#include <iostream>
TransEngine::TransEngine(int cache_size_per_block, const std::vector<std::pair<at::Tensor, at::Tensor>>& gpu_cache)
    : cache_size_per_block(cache_size_per_block), gpu_cache(gpu_cache){
    // Initialize parameters from config dictionaries
}

void TransEngine::recv_blocks(const std::string& channel, const std::string& request_id, const std::vector<uint32_t>& src_blocks, int opposite_rank, ncclComm_t& comm, c10::cuda::CUDAStream& stream) {

    c10::cuda::CUDAStreamGuard guard(stream);
    RecvBlocks(gpu_cache, src_blocks, cache_size_per_block, opposite_rank, comm);

    // at::cuda::CUDAEvent event;
    at::cuda::CUDAEvent* event = new at::cuda::CUDAEvent();

    event->record();

    if (recv_events.find(channel) == recv_events.end()) {
        recv_events[channel] = std::vector<std::pair<std::string, at::cuda::CUDAEvent*>>();
        recv_events[channel].push_back(std::make_pair(std::string(request_id), event));
    }
    else
        recv_events[channel].push_back(std::make_pair(request_id, event));
}

void TransEngine::send_blocks(const std::string& channel, const std::string& request_id, const std::vector<uint32_t>& dst_blocks, int opposite_rank, ncclComm_t& comm, c10::cuda::CUDAStream& stream) {

    c10::cuda::CUDAStreamGuard guard(stream);
    SendBlocks(gpu_cache, dst_blocks, cache_size_per_block, opposite_rank, comm);

    // at::cuda::CUDAEvent event;
    at::cuda::CUDAEvent* event = new at::cuda::CUDAEvent();
    event->record();
    if (send_events.find(channel) == send_events.end()) {
        send_events[channel] = std::vector<std::pair<std::string, at::cuda::CUDAEvent*>>();
        send_events[channel].push_back(std::make_pair(request_id, event));
    } else
        send_events[channel].push_back(std::make_pair(request_id, event));
}

void TransEngine::recv_layer_blocks(const std::string& channel, const std::string& request_id, const std::vector<uint32_t>& src_blocks, int opposite_rank, int layer, bool is_last_layer, ncclComm_t& comm, c10::cuda::CUDAStream& stream) {
    c10::cuda::CUDAStreamGuard guard(stream);
    std::cout<<"aaa"<<std::endl;
    RecvLayerBlocks(gpu_cache, src_blocks, cache_size_per_block, opposite_rank, layer, comm);
    std::cout<<"ccc"<<std::endl;
    if(is_last_layer) {
        at::cuda::CUDAEvent* event = new at::cuda::CUDAEvent();
        event->record();
        if (recv_events.find(channel) == recv_events.end()) {
            recv_events[channel] = std::vector<std::pair<std::string, at::cuda::CUDAEvent*>>();
            recv_events[channel].push_back(std::make_pair(std::string(request_id), event));
        }
        else
            recv_events[channel].push_back(std::make_pair(request_id, event));
    }
    std::cout<<"bbb"<<std::endl;
}

void TransEngine::send_layer_blocks(const std::string& channel, const std::string& request_id, const std::vector<uint32_t>& dst_blocks, int opposite_rank, int layer, bool is_last_layer, ncclComm_t& comm, c10::cuda::CUDAStream& stream) {

    c10::cuda::CUDAStreamGuard guard(stream);
    SendLayerBlocks(gpu_cache, dst_blocks, cache_size_per_block, opposite_rank, layer, comm);

    if(is_last_layer){
        at::cuda::CUDAEvent* event = new at::cuda::CUDAEvent();
        event->record();
        if (send_events.find(channel) == send_events.end()) {
            send_events[channel] = std::vector<std::pair<std::string, at::cuda::CUDAEvent*>>();
            send_events[channel].push_back(std::make_pair(request_id, event));
        } else
            send_events[channel].push_back(std::make_pair(request_id, event));
    }
}

std::vector<std::string> TransEngine::check_send_finished_events() {
    std::vector<std::string> send_blocks_finished;
    std::vector<std::string> finished_channels;

    for (auto& kv : send_events) {
        const std::string& channel = kv.first;
        auto& request_ids_and_events = kv.second;
        size_t num_finished_events = 0;

        for (auto it = request_ids_and_events.begin(); it != request_ids_and_events.end(); ++it) {
            const std::string& request_id = it->first;
            at::cuda::CUDAEvent *event = it->second;

            if (event->query()) {
                send_blocks_finished.emplace_back(TransferTaskMeta(channel, request_id).serialize());
                ++num_finished_events;
            } else {
                break;
            }
        }

        if (num_finished_events > 0) {
            // Remove finished events from the list
            request_ids_and_events.erase(request_ids_and_events.begin(), request_ids_and_events.begin() + num_finished_events);
        }
    }

    return send_blocks_finished;
}

std::vector<std::string> TransEngine::check_recv_finished_events() {
    std::vector<std::string> recv_blocks_finished;
    std::vector<std::string> finished_channels;

    for (auto& kv : recv_events) {
        const std::string& channel = kv.first;
        auto& request_ids_and_events = kv.second;
        size_t num_finished_events = 0;

        for (auto it = request_ids_and_events.begin(); it != request_ids_and_events.end(); ++it) {
            const std::string& request_id = it->first;
            at::cuda::CUDAEvent *event = it->second;

            if (event->query()) {
                recv_blocks_finished.emplace_back(TransferTaskMeta(channel, request_id).serialize());
                ++num_finished_events;
            } else {
                // std::cout<<"request_id not finished " << " " << request_id << std::endl;
                break;
            }
        }

        if (num_finished_events > 0) {
            // Remove finished events from the list
            request_ids_and_events.erase(request_ids_and_events.begin(), request_ids_and_events.begin() + num_finished_events);
        }
    }
    return recv_blocks_finished;
}

int TransEngine::create_nccl_comm(int32_t rank, ncclComm_t& comm, ncclUniqueId& uniqueId , int32_t NumDevice) {

    std::cout << "before create Global NCCL Comm " << rank << std::endl;
    ncclCommInitRank(&comm, NumDevice, uniqueId ,rank);
    std::cout << "Create Global NCCL Comm Success" << std::endl;
    return 0;
}


void TransEngine::RecvBlocks(std::vector<std::pair<at::Tensor, at::Tensor>>& dstCaches, \
    const std::vector<uint32_t>& dstBlocks, uint32_t cacheSize, uint32_t srcRank, ncclComm_t& comm)
{
    int layerNum = dstCaches.size();
    auto gpuStream = c10::cuda::getCurrentCUDAStream();
    auto cudaStream = gpuStream.stream();
    NCCLCHECK(ncclGroupStart());

    for (int i=0; i < layerNum; i++) {
        at::Tensor dstKeyCache = dstCaches[i].first;
        at::Tensor dstValueCache = dstCaches[i].second;

        for (int j = 0; j < dstBlocks.size(); j++) {
            int blockIdx = dstBlocks[j];
            void *dstKeyCachePtr = dstKeyCache.index({blockIdx}).data_ptr();
            void *dstValueCachePtr = dstValueCache.index({blockIdx}).data_ptr();
            if (ncclSuccess != ncclRecv(dstKeyCachePtr, cacheSize, ncclFloat, srcRank,\
                comm, cudaStream)) {
                std::cout << "[ERROR]  ncclRecv key cache error!!" << std::endl;
            }
            if (ncclSuccess != ncclRecv(dstValueCachePtr, cacheSize, ncclFloat, srcRank,\
                comm, cudaStream)) {
                std::cout << "[ERROR]  ncclRecv vaule cache error!!" << std::endl;
            }
        }
    }
    NCCLCHECK(ncclGroupEnd());
}


void TransEngine::SendBlocks(std::vector<std::pair<at::Tensor, at::Tensor>>& srcCaches, \
    const std::vector<uint32_t>& srcBlocks, uint32_t cacheSize, uint32_t destRank, ncclComm_t& comm)
{
    int layerNum = srcCaches.size();
    auto gpuStream = c10::cuda::getCurrentCUDAStream();
    auto cudaStream = gpuStream.stream();
    NCCLCHECK(ncclGroupStart());
    for (int i=0; i < layerNum; i++) {
        at::Tensor srcKeyCache = srcCaches[i].first;
        at::Tensor srcValueCache = srcCaches[i].second;

        for (int j = 0; j < srcBlocks.size(); j++) {
            int blockIdx = srcBlocks[j];
            void *srcKeyCachePtr = srcKeyCache.index({blockIdx}).data_ptr();
            void *srcValueCachePtr = srcValueCache.index({blockIdx}).data_ptr();
            if (ncclSuccess != ncclSend(srcKeyCachePtr, cacheSize, ncclFloat, destRank,\
                comm, cudaStream)) {
                std::cout << "[ERROR]  ncclSend key cache error!!" << std::endl;
            }
            if (ncclSuccess != ncclSend(srcValueCachePtr, cacheSize, ncclFloat, destRank,\
                comm, cudaStream)) {
                std::cout << "[ERROR]  ncclSend value cache error!!" << std::endl;
            }
        }
    }
    NCCLCHECK(ncclGroupEnd());
}

void TransEngine::SendLayerBlocks(std::vector<std::pair<at::Tensor, at::Tensor>>& srcCaches, \
    const std::vector<uint32_t>& srcBlocks, uint32_t cacheSize, uint32_t destRank, uint32_t layer, ncclComm_t& comm)
{
    auto gpuStream = c10::cuda::getCurrentCUDAStream();
    auto cudaStream = gpuStream.stream();
    NCCLCHECK(ncclGroupStart());
    at::Tensor srcKeyCache = srcCaches[layer].first;
    at::Tensor srcValueCache = srcCaches[layer].second;

    for (int j = 0; j < srcBlocks.size(); j++) {
        int blockIdx = srcBlocks[j];
        void *srcKeyCachePtr = srcKeyCache.index({blockIdx}).data_ptr();
        void *srcValueCachePtr = srcValueCache.index({blockIdx}).data_ptr();
        if (ncclSuccess != ncclSend(srcKeyCachePtr, cacheSize, ncclInt, destRank,\
            comm, cudaStream)) {
            std::cout << "[ERROR]  ncclSend key cache error!!" << std::endl;
        }
        if (ncclSuccess != ncclSend(srcValueCachePtr, cacheSize, ncclInt, destRank,\
            comm, cudaStream)) {
            std::cout << "[ERROR]  ncclSend value cache error!!" << std::endl;
        }
    }
    NCCLCHECK(ncclGroupEnd());
}

void TransEngine::RecvLayerBlocks(std::vector<std::pair<at::Tensor, at::Tensor>>& dstCaches, \
    const std::vector<uint32_t>& dstBlocks, uint32_t cacheSize, uint32_t srcRank, uint32_t layer, ncclComm_t& comm)
{
    auto gpuStream = c10::cuda::getCurrentCUDAStream();
    auto cudaStream = gpuStream.stream();
    NCCLCHECK(ncclGroupStart());
    at::Tensor dstKeyCache = dstCaches[layer].first;
    at::Tensor dstValueCache = dstCaches[layer].second;

    for (int j = 0; j < dstBlocks.size(); j++) {
        int blockIdx = dstBlocks[j];
        void *dstKeyCachePtr = dstKeyCache.index({blockIdx}).data_ptr();
        void *dstValueCachePtr = dstValueCache.index({blockIdx}).data_ptr();
        if (ncclSuccess != ncclRecv(dstKeyCachePtr, cacheSize, ncclFloat, srcRank,\
            comm, cudaStream)) {
            std::cout << "[ERROR]  ncclRecv key cache error!!" << std::endl;
        }
        if (ncclSuccess != ncclRecv(dstValueCachePtr, cacheSize, ncclFloat, srcRank,\
            comm, cudaStream)) {
            std::cout << "[ERROR]  ncclRecv vaule cache error!!" << std::endl;
        }
    }
    NCCLCHECK(ncclGroupEnd());
}