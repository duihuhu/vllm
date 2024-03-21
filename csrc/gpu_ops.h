#pragma once
#include <torch/extension.h>

using namespace torch::indexing;

int32_t CreateGlobalNcclComm(const char *rankTable, uint32_t globalRank);

void copy_blocks(std::vector<std::pair<at::Tensor, at::Tensor>> dstCaches, std::vector<std::pair<at::Tensor, at::Tensor>> srcCaches,
std::map<uint32_t, uint32_t> srcToDsts, uint32_t cacheSize, bool isCpu2Gpu);

void SendRequestRemote(uint64_t requestIdOnDevice, uint32_t requestIdSize, uint32_t destRank);

void RecvRequestRemote(uint64_t requestIdOnDevice, uint32_t requestIdSize, uint32_t srcRank);

void SendBlocksRemote(std::vector<std::pair<at::Tensor, at::Tensor>> srcCaches, \
    std::vector<uint32_t> srcBlocks, uint32_t cacheSize, uint32_t destRank);

void RecvBlocksRemote(std::vector<std::pair<at::Tensor, at::Tensor>> dstCaches, \
    std::vector<uint32_t> dstBlocks, uint32_t cacheSize, uint32_t srcRank);