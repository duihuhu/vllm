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
#include <chrono>

using namespace torch::indexing;
using namespace at; 

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


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

ncclComm_t g_globalNcclComm = nullptr;

int32_t CreateGlobalNcclComm(int32_t rank, int32_t NumDevice=8, int32_t size = 32) {
    constexpr int32_t ROOT_RANK = 0;
    constexpr int32_t TIME_OUT = 180;
    constexpr int32_t ROOT_INFO_OK = 1;
    constexpr const char *shmName = "NcclRootInfo";
    int32_t g_tpSize = NumDevice;
    if (NumDevice <=1) {
        g_tpSize = 1;
        return 0;
    }
    ncclUniqueId uniqueId;
    int shm_fd;
    int shmSize = sizeof(ncclUniqueId);
    std::cout << "Start rank " << rank << std::endl;
    if (rank == ROOT_RANK) {
        // 创建共享内存
        shm_fd = shm_open(shmName, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
        if (shm_fd < 0) {
            perror("shm_open");
            exit(1);
        }

        // 设置共享内存大小
        if (ftruncate(shm_fd, shmSize) == -1) {
            perror("ftruncate");
            exit(1);
        }

        // 映射共享内存
        void* shmaddr = mmap(NULL, shmSize, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (shmaddr == MAP_FAILED) {
            perror("mmap");
            exit(1);
        }

        // 生成唯一ID
        ncclGetUniqueId(&uniqueId);
        // char *out = (char*)(&uniqueId);
        // for(int i=0; i<shmSize; ++i)
        //     std::cout<<(out)[i];
        // std::cout<<std::endl;
        // 将唯一ID写入共享内存
        memcpy(shmaddr, &uniqueId, shmSize);

        // 解除映射
        if (munmap(shmaddr, shmSize) == -1) {
            perror("munmap");
            exit(1);
        }

    } else {
        int sleepTime = 0;
        // 等待共享内存就绪
        while (((shm_fd = shm_open(shmName, O_RDONLY, 0)) < 0) && (sleepTime < TIME_OUT)) {
            sleepTime++;
            sleep(1);
        }
        if (sleepTime >= TIME_OUT) {
            std::cout << "shm_open timeout" << std::endl;
            return -1;
        }
        // 映射共享内存
        void* shmaddr = mmap(NULL, shmSize, PROT_READ, MAP_SHARED, shm_fd, 0);
        if (shmaddr == MAP_FAILED) {
            perror("mmap");
            exit(1);
        }

        // 从共享内存中读取唯一ID
        memcpy(&uniqueId, shmaddr, shmSize);

        // char *out = (char*)(&uniqueId);
        // std::cout<<std::endl;
        // for(int i=0; i<shmSize; ++i)
        //     std::cout<<out[i];
        // std::cout<<std::endl;

        // 解除映射
        if (munmap(shmaddr, shmSize) == -1) {
            perror("munmap");
            exit(1);
        }
    }

    // 关闭共享内存
    // if (close(shm_fd) == -1) {
    //     perror("close");
    //     exit(1);
    // }
    std::cout << "Start init Global NCCL Comm Success" << std::endl;
    NCCLCHECK(ncclCommInitRank(&g_globalNcclComm, NumDevice, uniqueId ,rank));

    // 删除共享内存对象
    // if (shm_unlink(shmName) == -1) {
    //     perror("shm_unlink");
    //     exit(1);
    // }

    std::cout << "Create Global NCCL Comm Success" << std::endl;
    if (size!=0) {
        if (rank == 0){
            float *send_buf;
            auto gpuStream = c10::cuda::getCurrentCUDAStream();
            auto cudaStream = gpuStream.stream();
            cudaMalloc(&send_buf, size * sizeof(float));

            NCCLCHECK(ncclSend(send_buf, size , ncclFloat, 1, g_globalNcclComm, cudaStream));
            cudaStreamSynchronize(0);
        }
        else {
            float *recv_buf;
            auto gpuStream = c10::cuda::getCurrentCUDAStream();
            auto cudaStream = gpuStream.stream();
            cudaMalloc(&recv_buf, size * sizeof(float));

            NCCLCHECK(ncclRecv(recv_buf, size , ncclFloat, 0, g_globalNcclComm, cudaStream));
            cudaStreamSynchronize(0);
        }
    }
    return 0;
}

void copy_blocks_in_layer(std::vector<std::pair<at::Tensor, at::Tensor>> dstCaches, std::vector<std::pair<at::Tensor, at::Tensor>> srcCaches,\
std::map<uint32_t, uint32_t> srcToDsts, uint32_t cacheSize, bool isCpu2Gpu)
{
    using namespace torch::indexing;
    int layerNum = srcCaches.size();

    // int deviceId = 0;
    // aclrtGetDevice(&deviceId);
    // auto npuStream = c10_npu:getCurrentNPUStream(deviceId);
    // auto aclStream = npuStream.stream();

    int deviceId = 0;
    cudaGetDevice(&deviceId);
    auto gpuStream = c10::cuda::getCurrentCUDAStream();

    auto cudaStream = gpuStream.stream();
    CUDACHECK(cudaStreamCreate(&cudaStream));

    cudaMemcpyKind memcpy_type = isCpu2Gpu ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
    for (int i=0; i<layerNum; i++) {
        at::Tensor srcKeyCache = srcCaches[i].first;
        at::Tensor srcValueCache = srcCaches[i].second;

        at::Tensor dstKeyCache = dstCaches[i].first;
        at::Tensor dstValueCache = dstCaches[i].second;

        for (std::map<uint32_t, uint32_t>:: iterator it = srcToDsts.begin(); it != srcToDsts.end(); it++) {
            int src_idx = it->first;
            int dst_idx = it->second;
            void *dstKeyCachePtr = dstKeyCache.index({dst_idx}).data_ptr();
            void *srcKeyCachePtr = srcKeyCache.index({src_idx}).data_ptr();
            void *dstValueCachePtr = dstValueCache.index({dst_idx}).data_ptr();
            void *srcValueCachePtr = srcValueCache.index({src_idx}).data_ptr();

            if (cudaSuccess != cudaMemcpyAsync(dstKeyCachePtr, srcKeyCachePtr, cacheSize,\
                memcpy_type, cudaStream)) {
                    std::cout<< "[error] cudaMemcpy error!!" << std::endl;
            }

            if (cudaSuccess != cudaMemcpyAsync(dstValueCachePtr, srcValueCachePtr, cacheSize,\
                memcpy_type, cudaStream)) {
                    std::cout<< "[error] cudaMemcpy error!!" << std::endl;
            }

        }
    }
}

void SendRequestRemote(uint64_t requestIdOnDevice, uint32_t requestIdSize, uint32_t destRank)
{

    // int deviceId = 0;
    // cudaGetDevice(&deviceId);
    auto gpuStream = c10::cuda::getCurrentCUDAStream();
    auto cudaStream = gpuStream.stream();
    NCCLCHECK(ncclSend((void*) requestIdOnDevice, requestIdSize, ncclInt, destRank, g_globalNcclComm, cudaStream));

    return;
}

void RecvRequestRemote(uint64_t requestIdOnDevice, uint32_t requestIdSize, uint32_t srcRank)
{

    // int deviceId = 0;
    // cudaGetDevice(&deviceId);
    auto gpuStream = c10::cuda::getCurrentCUDAStream();
    auto cudaStream = gpuStream.stream();

    NCCLCHECK(ncclRecv((void*) requestIdOnDevice, requestIdSize, ncclInt, srcRank, g_globalNcclComm, cudaStream));

    return;
}

void SendBlocksRemote(std::vector<std::pair<at::Tensor, at::Tensor>> srcCaches, \
    std::vector<uint32_t> srcBlocks, uint32_t cacheSize, uint32_t destRank)
{
    int layerNum = srcCaches.size();

    // int deviceId = 0;
    // cudaGetDevice(&deviceId);
    auto gpuStream = c10::cuda::getCurrentCUDAStream();
    auto cudaStream = gpuStream.stream();
    // NCCLCHECK(ncclGroupStart());
    for (int i=0; i < layerNum; i++) {
        at::Tensor srcKeyCache = srcCaches[i].first;
        at::Tensor srcValueCache = srcCaches[i].second;

        for (int j = 0; j < srcBlocks.size(); j++) {
            int blockIdx = srcBlocks[j];
            void *srcKeyCachePtr = srcKeyCache.index({blockIdx}).data_ptr();
            void *srcValueCachePtr = srcValueCache.index({blockIdx}).data_ptr();
            // std::cout << "start send key cache: " << srcKeyCachePtr << std::endl;
            if (ncclSuccess != ncclSend(srcKeyCachePtr, cacheSize, ncclInt, destRank,\
                g_globalNcclComm, cudaStream)) {
                std::cout << "[ERROR]  ncclSend key cache error!!" << std::endl;
            }

            // std::cout << "start send value cache " << srcValueCachePtr << std::endl;

            if (ncclSuccess != ncclSend(srcValueCachePtr, cacheSize, ncclInt, destRank,\
                g_globalNcclComm, cudaStream)) {
                std::cout << "[ERROR]  ncclSend value cache error!!" << std::endl;
            }
        }
    }
    // NCCLCHECK(ncclGroupEnd());
    // std::cout << "send blocks success" << std::endl;
}

void RecvBlocksRemote(std::vector<std::pair<at::Tensor, at::Tensor>> dstCaches, \
    std::vector<uint32_t> dstBlocks, uint32_t cacheSize, uint32_t srcRank)
{
    int layerNum = dstCaches.size();

    // int deviceId = 0;
    // cudaGetDevice(&deviceId);
    auto gpuStream = c10::cuda::getCurrentCUDAStream();

    auto cudaStream = gpuStream.stream();
    // NCCLCHECK(ncclGroupStart());

    for (int i=0; i < layerNum; i++) {
        at::Tensor dstKeyCache = dstCaches[i].first;
        at::Tensor dstValueCache = dstCaches[i].second;

        for (int j = 0; j < dstBlocks.size(); j++) {
            int blockIdx = dstBlocks[j];
            void *dstKeyCachePtr = dstKeyCache.index({blockIdx}).data_ptr();
            void *dstValueCachePtr = dstValueCache.index({blockIdx}).data_ptr();
            // std::cout << "start recv key cache: " << dstKeyCachePtr << std::endl;
            if (ncclSuccess != ncclRecv(dstKeyCachePtr, cacheSize, ncclInt, srcRank,\
                g_globalNcclComm, cudaStream)) {
                std::cout << "[ERROR]  ncclRecv key cache error!!" << std::endl;
            }

            // std::cout << "start recv value cache " << dstValueCachePtr << std::endl;

            if (ncclSuccess != ncclRecv(dstValueCachePtr, cacheSize, ncclInt, srcRank,\
                g_globalNcclComm, cudaStream)) {
                std::cout << "[ERROR]  ncclRecv vaule cache error!!" << std::endl;
            }
        }
    }

    // NCCLCHECK(ncclGroupEnd());
    // std::cout << "recv blocks success" << std::endl;
}


void SendBlocksOnLayer(std::pair<at::Tensor, at::Tensor> srcCaches, \
    std::vector<uint32_t> srcBlocks, uint32_t cacheSize, uint32_t destRank)
{
    // int deviceId = 0;
    // cudaGetDevice(&deviceId);
    auto gpuStream = c10::cuda::getCurrentCUDAStream();
    auto cudaStream = gpuStream.stream();
    // NCCLCHECK(ncclGroupStart());

    at::Tensor srcKeyCache = srcCaches.first;
    at::Tensor srcValueCache = srcCaches.second;

    for (int j = 0; j < srcBlocks.size(); j++) {
        int blockIdx = srcBlocks[j];
        auto begin = std::chrono::steady_clock::now();
        auto timestamp_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(begin.time_since_epoch()).count();
        void *srcKeyCachePtr = srcKeyCache.index({blockIdx}).data_ptr();
        void *srcValueCachePtr = srcValueCache.index({blockIdx}).data_ptr();
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Send Copying time for buffer " << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;
        // std::cout << "start send key cache: " << srcKeyCachePtr << std::endl;
        if (ncclSuccess != ncclSend(srcKeyCachePtr, cacheSize, ncclInt, destRank,\
            g_globalNcclComm, cudaStream)) {
            std::cout << "[ERROR]  ncclSend key cache error!!" << std::endl;
        }

        // std::cout << "start send value cache " << srcValueCachePtr << std::endl;

        if (ncclSuccess != ncclSend(srcValueCachePtr, cacheSize, ncclInt, destRank,\
            g_globalNcclComm, cudaStream)) {
            std::cout << "[ERROR]  ncclSend value cache error!!" << std::endl;
        }

    }

    // NCCLCHECK(ncclGroupEnd());
    // std::cout << "send blocks success" << std::endl;
}


void HandleNcclCommDestroy()
{
    ncclCommDestroy(g_globalNcclComm);
}
