#include "trans_config.h"
#include "mooncake/common.h"
#include <iostream>
#include <fstream>
#include <memory>
std::string formatDeviceNames(const std::string& device_names) {
    std::stringstream ss(device_names);
    std::string item;
    std::vector<std::string> tokens;
    while (getline(ss, item, ',')) {
        tokens.push_back(item);
    }

    std::string formatted;
    for (size_t i = 0; i < tokens.size(); ++i) {
        formatted += "\"" + tokens[i] + "\"";
        if (i < tokens.size() - 1) {
            formatted += ",";
        }
    }
    return formatted;
}

std::string loadNicPriorityMatrix(const std::string& mc_device_name, const std::string& mc_nic_proirity_matrix) {
    if (!mc_nic_proirity_matrix.empty()) {
        std::ifstream file(mc_nic_proirity_matrix);
        if (file.is_open()) {
            std::string content((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
            file.close();
            return content;
        }
    }
    // Build JSON Data
    auto device_names = formatDeviceNames(mc_device_name);
    return "{\"cpu:0\": [[" + device_names + "], []], "
           " \"cpu:1\": [[" + device_names + "], []], "
           " \"gpu:0\": [[" + device_names + "], []]}";
}

TransManager::TransManager(
    int cache_size_per_block,
    std::vector<std::pair<at::Tensor, at::Tensor>> &gpu_cache, int rank,
    int local_rank, int nccl_local_rank, int tp, int num_layer,
    int cache_block_size, std::vector<uint64_t> &blocks_gpu_cache,
    const std::string &mc_local_server_name,
    const std::string &mc_metadata_server, const std::string &mc_device_name,
    const std::string &mc_nic_proirity_matrix, const std::string &mc_protocol,
    const std::map<int, std::string> &mc_servers_addr,
    std::vector<std::pair<at::Tensor, at::Tensor>> &remote_swap_cpu_cache,
    std::vector<uint64_t> &remote_swap_blocks_address)
    : cache_size_per_block(cache_size_per_block), gpu_cache(gpu_cache),
      rank(rank), local_rank(local_rank), nccl_local_rank(nccl_local_rank),
      tp(tp), num_layer(num_layer), cache_block_size(cache_block_size),
      blocks_gpu_cache(blocks_gpu_cache),
      mc_servers_addr_(mc_servers_addr),
      remote_swap_cpu_cache_(remote_swap_cpu_cache),
      remote_swap_blocks_address_(remote_swap_blocks_address) {
  auto metadata =
      std::make_shared<mooncake::TransferMetadata>(mc_metadata_server, "redis");
  if (!metadata) {
    throw std::runtime_error("construct TransferMetadata error");
  }
  transfer_engine_ = std::make_shared<mooncake::TransferEngine>(metadata);
  if (!transfer_engine_) {
    throw std::runtime_error("construct TransferEngine error");
  }
  auto hostname_port = mooncake::parseHostNameWithPort(mc_local_server_name.c_str());
  int ret = transfer_engine_->init(mc_local_server_name.c_str(), hostname_port.first.c_str(),
                                   hostname_port.second);
  if (ret) {
    throw std::runtime_error("TransferEngine init error");
  }
  if (mc_protocol == "rdma") {
    auto nic_priority_matrix =
        loadNicPriorityMatrix(mc_device_name, mc_nic_proirity_matrix);
    void **args = (void **)malloc(2 * sizeof(void *));
    args[0] = (void *)nic_priority_matrix.c_str();
    args[1] = nullptr;
    xport_ =
        std::shared_ptr<mooncake::Transport>(transfer_engine_->installOrGetTransport("rdma", args));
  } else if (mc_protocol == "tcp") {
    xport_ = std::shared_ptr<mooncake::Transport>(
        transfer_engine_->installOrGetTransport("tcp", nullptr));
  } else {
    std::cout << "Unsupported protocol" << std::endl;
  }

  if (!xport_) {
    throw std::runtime_error("construct Transport error");
  }
  if(remote_swap_cpu_cache_.size() > 1 && remote_swap_blocks_address_.size() > 0) {
    throw std::runtime_error("size of remote_swap_cpu_cache_ > 1 and size of remote_swap_blocks_address_ > 0 cannot appear simultaneously");
  }

  // register gpu
  if(gpu_cache.size() > 1 && blocks_gpu_cache.size() > 0) {
    throw std::runtime_error("size of gpu_cache > 1 and size of blocks_gpu_cache > 0 cannot appear simultaneously");
    
  }
  if (gpu_cache.size() > 1) {
    std::string location = std::string("gpu:") + std::to_string(local_rank);
    for(auto& kv_tensor: gpu_cache) {
       auto res = transfer_engine_->registerLocalMemory(kv_tensor.first.data_ptr(), kv_tensor.first.nbytes(), location, true, true);
       if(res != 0) {
        throw std::runtime_error("register gpu memory of key error in block level");
       }
       res = transfer_engine_->registerLocalMemory(kv_tensor.second.data_ptr(), kv_tensor.second.nbytes(), location, true, true);
       if(res != 0) {
        throw std::runtime_error("register gpu memory of value error in block level");
       }
       mc_num_gpu_bufs_ += 2;
    }
  }
  if(blocks_gpu_cache.size() > 0) {
    std::string location = std::string("gpu:") + std::to_string(local_rank);
    for(auto& block_address: blocks_gpu_cache) {
        auto res = transfer_engine_->registerLocalMemory((void *)block_address, cache_block_size, location, true, true);
        if(res != 0) {
            throw std::runtime_error("register gpu memory of key error in agg_block");
        }
        mc_num_gpu_bufs_ += 1;
    }
  }
  // register cpu
  if(remote_swap_cpu_cache_.size() > 1) {
    std::string location = std::string("cpu:0");
    for(auto& kv_tensor: remote_swap_cpu_cache_) {
       auto res = transfer_engine_->registerLocalMemory(kv_tensor.first.data_ptr(), kv_tensor.first.nbytes(), location, true, true);
       if(res != 0) {
        throw std::runtime_error("register cpu memory of key error in block level");
       }
       res = transfer_engine_->registerLocalMemory(kv_tensor.second.data_ptr(), kv_tensor.second.nbytes(), location, true, true);
       if(res != 0) {
        throw std::runtime_error("register cpu memory of value error in block level");
       }
    }
  }
  if (remote_swap_blocks_address_.size() > 0) {
    std::string location = std::string("cpu:0");
    for(auto& block_address: remote_swap_blocks_address_) {
        auto res = transfer_engine_->registerLocalMemory((void *)block_address, cache_block_size, location, true, true);
        if(res != 0) {
            throw std::runtime_error("register cpu memory of key error in agg_block");
        }
    }
  }

  std::cout << "[RANK " << nccl_local_rank
            << "] init TransferEngine, device: " << mc_device_name << std::endl;
  execute = std::thread(&TransManager::dist_worker, this);
}

TransManager::~TransManager() {
    if (execute.joinable()) {
        execute.join();
    }
}
void TransManager::dist_worker() {
    while (true) {
        if(!worker_task_queue.empty()) {
            auto worker_task = worker_task_queue.pop_front();
            TaskType task_type = worker_task.type;
            TransWorker* task_worker = nullptr;
            switch (task_type) {
                case TaskType::TRANSFER_SEND_BLOCKS:
                    task_worker = send_trans_workers[worker_task.meta.channel];
                    task_worker->add_tasks(worker_task);
                    break;
                case TaskType::TRANSFER_RECV_BLOCKS:
                    task_worker = recv_trans_workers[worker_task.meta.channel];
                    task_worker->add_tasks(worker_task);
                    break;
                case TaskType::TRANSFER_SEND_LAYER_BLOCKS:
                    task_worker = send_trans_workers[worker_task.meta.channel];
                    task_worker->add_tasks(worker_task);
                    break;
                case TaskType::TRANSFER_RECV_LAYER_BLOCKS:
                    task_worker = recv_trans_workers[worker_task.meta.channel];
                    task_worker->add_tasks(worker_task);
                    break;
                case TaskType::TRANSFER_SEND_FULL_BLOCKS:
                    task_worker = send_trans_workers[worker_task.meta.channel];
                    task_worker->add_tasks(worker_task);
                    break;
                case TaskType::TRANSFER_RECV_FULL_BLOCKS:
                    task_worker = recv_trans_workers[worker_task.meta.channel];
                    task_worker->add_tasks(worker_task);
                    break;
                case TaskType::TRANSFER_HBM_TO_DRAM_BLOCKS:
                    task_worker = swap_workers[worker_task.meta.channel];
                    task_worker->add_tasks(worker_task);
                    break;
                case TaskType::TRANSFER_HBM_TO_DRAM_FULL_BLOCKS:
                    task_worker = swap_workers[worker_task.meta.channel];
                    task_worker->add_tasks(worker_task);
                    break;
                default:
                    throw std::runtime_error("invalid task_type.");
            }
        }
    }
    return;
}

std::vector<char> TransManager::get_nccl_id(const std::string& dst_channel, const std::string& worker_type){
    ncclUniqueId uniqueId; 
    NCCLCHECK(ncclGetUniqueId(&uniqueId));
    if(worker_type=="sender"){
        if(send_trans_workers.find(dst_channel) == send_trans_workers.end()){
            TransWorker* task_worker = new TransWorker(cache_size_per_block, gpu_cache, rank, local_rank, nccl_local_rank, dst_channel, tp, num_layer, cache_block_size, blocks_gpu_cache, transfer_engine_, xport_, mc_servers_addr_, mc_num_gpu_bufs_);
            send_trans_workers[dst_channel] = task_worker;
        }
    } else{
        if(recv_trans_workers.find(dst_channel) == recv_trans_workers.end()){
            TransWorker* task_worker = new TransWorker(cache_size_per_block, gpu_cache, rank, local_rank, nccl_local_rank, dst_channel, tp, num_layer, cache_block_size, blocks_gpu_cache, transfer_engine_, xport_, mc_servers_addr_, mc_num_gpu_bufs_);
            recv_trans_workers[dst_channel] = task_worker;
        }
    }
    return std::vector<char>(uniqueId.internal, uniqueId.internal + sizeof(uniqueId.internal));
}

void TransManager::add_tasks(const std::vector<std::string>& tasks) {
    for (const auto& task : tasks) {
        auto trans_task = TransferTask::deserialize(task);
        worker_task_queue.push_back(trans_task);
    }
}
void TransManager::create_comm(std::vector<char>& nccl_id ,const std::string& dst_channel, const std::string& worker_type){
    if(worker_type=="sender"){
        if(send_trans_workers.find(dst_channel) == send_trans_workers.end()){
            TransWorker* task_worker = new TransWorker(cache_size_per_block, gpu_cache, rank, local_rank, nccl_local_rank, dst_channel, tp, num_layer, cache_block_size, blocks_gpu_cache, transfer_engine_, xport_, mc_servers_addr_, mc_num_gpu_bufs_);
            send_trans_workers[dst_channel] = task_worker;
        }
        TransWorker* task_worker = send_trans_workers[dst_channel];
        task_worker->add_comm_task(nccl_id);
    } else{
        if(recv_trans_workers.find(dst_channel) == recv_trans_workers.end()){
            TransWorker* task_worker = new TransWorker(cache_size_per_block, gpu_cache, rank, local_rank, nccl_local_rank, dst_channel, tp, num_layer, cache_block_size, blocks_gpu_cache, transfer_engine_, xport_, mc_servers_addr_, mc_num_gpu_bufs_);
            recv_trans_workers[dst_channel] = task_worker;
        }
        TransWorker* task_worker = recv_trans_workers[dst_channel];
        task_worker->add_comm_task(nccl_id);
    }
    return;
}

void TransManager::init_dst_cpu_cache(const std::string& dst_channel, const std::vector<std::pair<at::Tensor, at::Tensor>>& dst_cpu_cache, const std::vector<uint64_t>& dst_blocks_cpu_cache) {
    if(swap_workers.find(dst_channel) == swap_workers.end()){
        std::cout << "dst_channel " << dst_channel << " " << nccl_local_rank << std::endl;
        // TransWorker* task_worker = new TransWorker(cache_size_per_block, gpu_cache, rank, local_rank, nccl_local_rank, dst_channel, tp, num_layer, cache_block_size, blocks_gpu_cache, dst_cpu_cache);
        TransWorker* task_worker = new TransWorker(cache_size_per_block, gpu_cache, rank, local_rank, nccl_local_rank, dst_channel, tp, num_layer, cache_block_size, blocks_gpu_cache, dst_cpu_cache, dst_blocks_cpu_cache, transfer_engine_, xport_, mc_servers_addr_, mc_num_gpu_bufs_);

        swap_workers[dst_channel] = task_worker;
    }
    return;
}

std::vector<std::vector<std::tuple<std::vector<std::string>, std::vector<std::string>,std::vector<std::string>>>>TransManager::get_finished_transfer_tasks() {
    std::vector<std::vector<std::tuple<std::vector<std::string>, std::vector<std::string>,std::vector<std::string>>>> finished_work_tasks;
    for (const auto& pair : send_trans_workers) {
        // const std::string& key = pair.first;
        TransWorker* worker = pair.second;
        auto finished_work_task = worker->get_finished_transfer_tasks();
        if(!finished_work_task.empty()) {
            finished_work_tasks.emplace_back(finished_work_task);
        }
    }
    for (const auto& pair : recv_trans_workers) {
        // const std::string& key = pair.first;
        TransWorker* worker = pair.second;
        auto finished_work_task = worker->get_finished_transfer_tasks();
        if(!finished_work_task.empty()) {
            finished_work_tasks.emplace_back(finished_work_task);
        }
    }

    for (const auto& pair : swap_workers) {
        TransWorker* worker = pair.second;
        auto finished_work_task = worker->get_finished_transfer_tasks();
        if(!finished_work_task.empty()) {
            finished_work_tasks.emplace_back(finished_work_task);
        }
    }

    return finished_work_tasks;
}
