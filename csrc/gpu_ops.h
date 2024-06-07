#pragma once
#include <torch/extension.h>
#include <cassert>
#include <iostream>
#include <torch/extension.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include "nccl.h"
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <sys/time.h>
#include <chrono>

using namespace torch::indexing;

int32_t CreateGlobalMulNcclComm(int32_t rank, int32_t NumDevice , int32_t num_comms = 4);

int32_t CreateGlobalNcclComm(int32_t rank, int32_t NumDevice=8, int32_t num_comms = 16);

void copy_blocks_in_layer(std::vector<std::pair<at::Tensor, at::Tensor>> dstCaches, std::vector<std::pair<at::Tensor, at::Tensor>> srcCaches,
std::map<uint32_t, uint32_t> srcToDsts, uint32_t cacheSize, bool isCpu2Gpu);

void SendRequestRemote(uint64_t requestIdOnDevice, uint32_t requestIdSize, uint32_t destRank);

void RecvRequestRemote(uint64_t requestIdOnDevice, uint32_t requestIdSize, uint32_t srcRank);

void SendBlocksRemote(std::vector<std::pair<at::Tensor, at::Tensor>> srcCaches, \
    std::vector<uint32_t> srcBlocks, uint32_t cacheSize, uint32_t destRank);

void RecvBlocksRemote(std::vector<std::pair<at::Tensor, at::Tensor>> dstCaches, \
    std::vector<uint32_t> dstBlocks, uint32_t cacheSize, uint32_t srcRank);

void SendLayerBlocks(std::vector<std::pair<at::Tensor, at::Tensor>> srcCaches, \
    std::vector<uint32_t> srcBlocks, uint32_t cacheSize, uint32_t destRank, uint32_t layer);

void RecvLayerBlocks(std::vector<std::pair<at::Tensor, at::Tensor>> dstCaches, \
    std::vector<uint32_t> dstBlocks, uint32_t cacheSize, uint32_t destRank, uint32_t layer);
void HandleNcclCommDestroy();