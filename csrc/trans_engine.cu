#include "trans_config.h"
#include <stdexcept>
#include <iostream>

TransEngine::TransEngine(const TransConfig& trans_config, const std::vector<std::pair<at::Tensor, at::Tensor>>& gpu_cache)
    : trans_config(trans_config), gpu_cache(gpu_cache){
    // Initialize parameters from config dictionaries
}

void TransEngine::recv_blocks(const std::string& channel, const std::string& request_id, const std::vector<int>& src_blocks, int opposite_rank) {
    if (recv_streams.find(channel) == recv_streams.end()) {
        // c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool(true);
        c10::cuda::CUDAStream* stream = new c10::cuda::CUDAStream(c10::cuda::getStreamFromPool(true));
        recv_streams[channel] = stream;
    }
    
    c10::cuda::CUDAStreamGuard guard(recv_streams[channel]);
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
}
void TransEngine::send_blocks(const std::string& channel, const std::string& request_id, const std::vector<int>& dst_blocks, int opposite_rank) {
    if (send_streams.find(channel) == send_streams.end()) {
        // c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool(true);
        c10::cuda::CUDAStream* stream = new c10::cuda::CUDAStream(c10::cuda::getStreamFromPool(true));
        send_streams[channel] = stream;
    }

    c10::cuda::CUDAStreamGuard guard(send_streams[channel]);
    SendBlocksRemote(gpu_cache, dst_blocks, cache_size_per_block, opposite_rank);

    // at::cuda::CUDAEvent event;
    at::cuda::CUDAEvent* event = new at::cuda::CUDAEvent();
    event->record();

    if (send_events.find(channel) == send_events.end()) {
        send_events[channel] = std::vector<std::pair<std::string, at::cuda::CUDAEvent*>>();
        recv_events[channel].push_back(std::make_pair(request_id, event));
    } else
        send_events[channel].push_back(std::make_pair(request_id, event));
}

std::vector<TransferTaskMeta> TransEngine::check_send_finished_events() {
    std::vector<TransferTaskMeta> send_blocks_finished;
    std::vector<std::string> finished_channels;

    for (auto& kv : send_events) {
        const std::string& channel = kv.first;
        auto& request_ids_and_events = kv.second;
        size_t num_finished_events = 0;

        for (auto it = request_ids_and_events.begin(); it != request_ids_and_events.end(); ++it) {
            const std::string& request_id = it->first;
            at::cuda::CUDAEvent *event = it->second;

            if (event->query()) {
                send_blocks_finished.emplace_back(TransferTaskMeta(channel, request_id));
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

std::vector<TransferTaskMeta> TransEngine::check_recv_finished_events() {
    std::vector<TransferTaskMeta> recv_blocks_finished;
    std::vector<std::string> finished_channels;

    for (auto& kv : recv_events) {
        const std::string& channel = kv.first;
        auto& request_ids_and_events = kv.second;
        size_t num_finished_events = 0;

        for (auto it = request_ids_and_events.begin(); it != request_ids_and_events.end(); ++it) {
            const std::string& request_id = it->first;
            at::cuda::CUDAEvent *event = it->second;

            if (event->query()) {
                recv_blocks_finished.emplace_back(TransferTaskMeta(channel, request_id));
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

    return recv_blocks_finished;
}