#include "trans_config.h"
#include <stdexcept>
#include <iostream>
TransEngine::TransEngine(int cache_size_per_block, const std::vector<std::pair<at::Tensor, at::Tensor>>& gpu_cache)
    : cache_size_per_block(cache_size_per_block), gpu_cache(gpu_cache), num_stream(0){
    // Initialize parameters from config dictionaries
    for (int i = 0; i < 4; ++i) {
         streams.push_back(c10::cuda::CUDAStream(c10::cuda::getStreamFromPool(true)));
    }
}

void TransEngine::recv_blocks(const std::string& channel, const std::string& request_id, const std::vector<uint32_t>& src_blocks, int opposite_rank) {

    c10::cuda::CUDAStream& stream = streams[num_stream];
    c10::cuda::CUDAStreamGuard guard(stream);
    RecvBlocksRemote(gpu_cache, src_blocks, cache_size_per_block, opposite_rank);

    // at::cuda::CUDAEvent event;
    at::cuda::CUDAEvent* event = new at::cuda::CUDAEvent();

    event->record();

    if (recv_events.find(channel) == recv_events.end()) {
        recv_events[channel] = std::vector<std::pair<std::string, at::cuda::CUDAEvent*>>();
        recv_events[channel].push_back(std::make_pair(std::string(request_id), event));
    }
    else
        recv_events[channel].push_back(std::make_pair(request_id, event));
    num_stream = (num_stream + 1) % 4;
}

void TransEngine::recv_layer_blocks(const std::string& channel, const std::string& request_id, const std::vector<uint32_t>& src_blocks, int opposite_rank, int num_layer) {

    for(int layer = 0; layer < num_layer; layer++) {
        c10::cuda::CUDAStream& stream = streams[num_stream];
        c10::cuda::CUDAStreamGuard guard(stream);
        RecvLayerBlocks(gpu_cache, src_blocks, cache_size_per_block, opposite_rank, layer);
        if(layer == num_layer - 1) {
            at::cuda::CUDAEvent* event = new at::cuda::CUDAEvent();
            event->record();
            if (recv_events.find(channel) == recv_events.end()) {
                recv_events[channel] = std::vector<std::pair<std::string, at::cuda::CUDAEvent*>>();
                recv_events[channel].push_back(std::make_pair(std::string(request_id), event));
            }
            else
                recv_events[channel].push_back(std::make_pair(request_id, event));
        }
        num_stream = (num_stream + 1) % 4;
    }
}

void TransEngine::send_layer_blocks(const std::string& channel, const std::string& request_id, const std::vector<uint32_t>& dst_blocks, int opposite_rank, int layer, bool is_last_layer) {

    // c10::cuda::CUDAStreamGuard guard(*send_streams[channel]);
    c10::cuda::CUDAStream& stream = streams[num_stream];
    c10::cuda::CUDAStreamGuard guard(stream);
    SendLayerBlocks(gpu_cache, dst_blocks, cache_size_per_block, opposite_rank, layer);

    // at::cuda::CUDAEvent event;
    if(is_last_layer){
        // std::cout<<"send_layer_blocks is_last_layer is end " << layer << " "<< is_last_layer << " " << request_id << std::endl;
        at::cuda::CUDAEvent* event = new at::cuda::CUDAEvent();
        event->record();
        if (send_events.find(channel) == send_events.end()) {
            send_events[channel] = std::vector<std::pair<std::string, at::cuda::CUDAEvent*>>();
            send_events[channel].push_back(std::make_pair(request_id, event));
        } else
            send_events[channel].push_back(std::make_pair(request_id, event));
    }
    num_stream = (num_stream + 1) % 4;
}


void TransEngine::send_blocks(const std::string& channel, const std::string& request_id, const std::vector<uint32_t>& dst_blocks, int opposite_rank) {
    // c10::cuda::CUDAStreamGuard guard(*send_streams[channel]);
    c10::cuda::CUDAStream& stream = streams[num_stream];
    c10::cuda::CUDAStreamGuard guard(stream);

    SendBlocksRemote(gpu_cache, dst_blocks, cache_size_per_block, opposite_rank);

    // at::cuda::CUDAEvent event;
    at::cuda::CUDAEvent* event = new at::cuda::CUDAEvent();
    event->record();
    if (send_events.find(channel) == send_events.end()) {
        send_events[channel] = std::vector<std::pair<std::string, at::cuda::CUDAEvent*>>();
        send_events[channel].push_back(std::make_pair(request_id, event));
    } else
        send_events[channel].push_back(std::make_pair(request_id, event));
    num_stream = (num_stream + 1) % 4;

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