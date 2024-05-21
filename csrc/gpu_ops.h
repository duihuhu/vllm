#pragma once
#include <torch/extension.h>

using namespace torch::indexing;

int32_t CreateGlobalNcclComm(int32_t rank, int32_t NumDevice=8, int32_t size = 32);

void copy_blocks_in_layer(std::vector<std::pair<at::Tensor, at::Tensor>> dstCaches, std::vector<std::pair<at::Tensor, at::Tensor>> srcCaches,
std::map<uint32_t, uint32_t> srcToDsts, uint32_t cacheSize, bool isCpu2Gpu);

void SendRequestRemote(uint64_t requestIdOnDevice, uint32_t requestIdSize, uint32_t destRank);

void RecvRequestRemote(uint64_t requestIdOnDevice, uint32_t requestIdSize, uint32_t srcRank);

void SendBlocksRemote(std::vector<std::pair<at::Tensor, at::Tensor>> srcCaches, \
    std::vector<uint32_t> srcBlocks, uint32_t cacheSize, uint32_t destRank);

void RecvBlocksRemote(std::vector<std::pair<at::Tensor, at::Tensor>> dstCaches, \
    std::vector<uint32_t> dstBlocks, uint32_t cacheSize, uint32_t srcRank);

void SendBlocksOnLayer(std::pair<at::Tensor, at::Tensor> srcCaches, \
    std::vector<uint32_t> srcBlocks, uint32_t cacheSize, uint32_t destRank);

void SendBlockOnLayer(uint32_t k_addr, uint32_t v_addr, uint32_t cacheSize, uint32_t destRank);

void HandleNcclCommDestroy();