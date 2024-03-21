#include <cassert>
#include <iostream>
#include <torch/extension.h>
// #include <torch_npu/csrc/core/npu/NPUStream.h>
// #include "acl/acl.h"
//#include "kernel_entry.h"
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include "nccl.h"
#include <ATen/cuda/CUDAStream.h>
#include <cuda_runtime.h>
// #include "mpi.h"

//#include "hccl/hccl.h"
//#include "hccl/hccl_type.h"
using namespace torch::indexing;

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

//HcclComm g_globalHcclComm = nullptr;

// int32_t CreateGlobalNcclComm(const char *rankTable, uint32_t globalRank) 
// {
//     if (ncclSuccess != NcclCommInitClusterInfo(rankTable, globalRank, &g_globalNcclComm)) {
//         return -1;
//     }
//     std::cout << "create global hccl comm success" << std::endl;
//     return 0;
// }

int32_t CreateGlobalNcclComm(int32_t rank, int32_t NumDevice=8) {
    constexpr int32_t ROOT_RANK = 0;
    constexpr int32_t TIME_OUT = 180;
    constexpr int32_t ROOT_INFO_OK = 1;
    constexpr const char *shmName = "NcclRootInfo";
    int32_t g_tpSize = NumDevice;
    if (NumDevice < =1) {
        g_tpSize = 1;
        return 0;
    }
    ncclUniqueId uniqueId;
    int shm_fd;
    int shmSize = sizeof(ncclUniqueId);
    if (rank == ROOT_RANK) {
        // 创建共享内存
        shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
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
        while (((shm_fd = shm_open(SHM_NAME, O_RDONLY, 0)) < 0) && (sleepTime < TIME_OUT)) {
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

        // 解除映射
        if (munmap(shmaddr, shmSize) == -1) {
            perror("munmap");
            exit(1);
        }
    }

    // 关闭共享内存
    if (close(shm_fd) == -1) {
        perror("close");
        exit(1);
    }

    NCCLCHECK(ncclCommInitRank(&g_globalNcclComm, NumDevice, uniqueId ,rank))

    // 删除共享内存对象
    if (shm_unlink(SHM_NAME) == -1) {
        perror("shm_unlink");
        exit(1);
    }

    std::cout << "Create Global NCCL Comm Success" << std::endl;

    return 0;
}

// int32_t CreateGlobalNcclComm(int32_t rank, int32_t NumDevice=8) {
//     constexpr int32_t ROOT_RANK = 0;
//     constexpr int32_t TIME_OUT = 180;
//     constexpr int32_t ROOT_INFO_OK = 1;
//     constexpr const char *shmName = "NcclRootInfo";
//     g_tpSize = tpSize;
//     if (tpSize < =1) {
//         g_tpSize = 1;
//         return 0;
//     }

//     ncclRootInfo *pRootInfo = nullptr;
//     int shm_fd;
//     int shmSize = sizeof(ncclRootInfo) + sizeof(int32_t);
//     if (rank == ROOT_RANK) {
//         /* create the shared memory object */
//         shm_fd = shm_open(shmName, O_CREAT | O_RDWR, 0666);
//         /*configure the size of the shared memory object*/
//         ftruncate(shm_df, shmSize);
//         /* memory map the shared memory object*/
//         pRootInfo = (ncclRootInfo *) mmap(0, shmSize, PROT_WRITE, MAP_SHARED, shm_fd, 0);

//         for (int i=0; i<shmSize; i++) {
//             ((char *)pRootInfo)[i] = 0;
//         }

//         // HCCLCHECK(ncclGetRootInfo(pRootInfo));
        
//         // End tag of RootInfo
//         *(int32_t *)((char *)pRootInfo + sizeof(ncclRootInfo)) = ROOT_INFO_OK;
//     } else {
//         int sleepTime = 0;
//         /* open the shared memory object */
//         while (((shm_fd = shm_open(shmName, O_RDONLY, 0666)) < 0) && (sleepTime < TIME_OUT)) {
//             sleepTime++;
//             sleep(1);
//         }
//         if (sleepTime >= TIME_OUT) {
//             std::cout << "shm_open timeout" << std::endl;
//             return -1;
//         }

//         /*memory map the shared memory object */
//         if ((pRootInfo = (ncclRootInfo *)mmap(0, shmSize, PROT_READ, MAP_SHARED, shm_fd, 0)) == nullptr) {
//             std::cout << "mmap Error" << std::endl;
//             return -1;
//         }

//         sleepTime = 0;
//         while ((*(int32_t *)((char *)pRootInfo + sizeof(ncclRootInfo)) != ROOT_INFO_OK) && (sleepTime < TIME_OUT)) {
//             sleepTime++;
//             sleep(1);
//         }

//         if (sleepTime >= TIME_OUT) {
//             std::cout << "shm_open timeout" << std::endl;
//             return -1;
//         }
//     }

//     // HCCLCHECK(HcclCommInitRootInfo(tpSize, pRootInfo, rank, &g_globalNcclComm));

//     NCCLCHECK(ncclCommInitRank(&g_globalNcclComm, tpRank, uniqueId ,rank)
//     /*remote the shared memory object*/
//     shm_unlink(shmName);
//     std::cout << "Create Tp Hccl Comm Success" << std::endl;

//     return 0;

// }

void copy_blocks_in_layer(std::vector<std::pair<at::Tensor, at::Tensor>> dstCaches, std::vector<std::pair<at::Tensor, at::Tensor>> srcCaches,
std:map<uint32_t, uint32_t> srcToDsts, uint32_t cacheSize, bool isCpu2Gpu)
{
    using namespace torch::indexing;
    int layerNum = srcCaches.size();

    // int deviceId = 0;
    // aclrtGetDevice(&deviceId);
    // auto npuStream = c10_npu:getCurrentNPUStream(deviceId);
    // auto aclStream = npuStream.stream();

    int deviceId = 0;
    cudaGetDevice(&deviceId);
    auto gpuStream = at::cuda::getCurrentCUDAStream();
    auto cudaStream = gpuStream.stream();
    CUDACHECK(cudaStreamCreate(&cudaStream));

    cudaMemcpyKind memcpy_type = isCpu2Gpu ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
    for (int i=0; i<layerNum; i++) {
        at::Tensor srcKeyCache = srcCaches[i].first;
        at::Tensor srcValueCache = srcCaches[i].second;

        at::Tensor dstKeyCache = dstCaches[i].first;
        at::Tensor dstValueCache = dstCaches[i].second;

        for (std::map<<uint32_t, uint32_t>:: iterator it = srcToDsts.begin(); it != srcToDsts.end(); it++) {
            int src_idx = it->first;
            int dst_idx = it->second;
            void *dstKeyCachePtr = dstKeyCache.index({dst_idx}).data_ptr();
            void *srcKeyCachePtr = srcKeyCache.index({src_idx}).data_ptr();
            void *dstValueCachePtr = dstValueCache.index({dst_idx}).data_ptr();
            void *srcValueCachePtr = srcValueCache.index({src_idx}).data_ptr();

            if (ACL_SUCCESS != cudaMemcpyAsync(dstKeyCachePtr, cacheSize, srcKeyCachePtr, cacheSize,\
                memcpy_type, cudaStream)) {
                    std::cout<< "[error] aclrMemcpy error!!" << std::endl;
            }

            if (ACL_SUCCESS != cudaMemcpyAsync(dstValueCachePtr, cacheSize, srcValueCachePtr, cacheSize,\
                memcpy_type, cudaStream)) {
                    std::cout<< "[error] aclrMemcpy error!!" << std::endl;
            }

        }
    }
}

void SendRequestRemote(uint64_t requestIdOnDevice, uint32_t requestIdSize, uint32_t destRank)
{
    // int deviceId = 0;
    // aclrtGetDevice(&deviceId);
    // auto npuStream = c10_npu:getCurrentNPUStream(deviceId);
    // auto aclStream = npuStream.stream();

    // HcclResult code = HcclSend((void*) requestIdOnDevice, requestIdSize, HCCL_DATA_TYPE_INT8, \
    //     destRank, g_globalHcclComm, aclStream);
    
    // if (HCCL_SUCCESS != code) {
    //     std::cout << "ERROR HcclSend request id error" << code << std::endl;
    // }
    int deviceId = 0;
    cudaGetDevice(&deviceId);
    auto gpuStream = at::cuda::getCurrentCUDAStream();
    auto cudaStream = gpuStream.stream();
    NCCLCHECK(ncclSend((void*) requestIdOnDevice, requestIdSize, ncclInt, destRank, g_globalNcclComm, cudaStream));
    // Synchronize stream
    // CUDACHECK(cudaStreamSynchronize(cudaStream));

    return;
}

void RecvRequestRemote(uint64_t requestIdOnDevice, uint32_t requestIdSize, uint32_t srcRank)
{
    // int deviceId = 0;
    // aclrtGetDevice(&deviceId);
    // auto npuStream = c10_npu:getCurrentNPUStream(deviceId);
    // auto aclStream = npuStream.stream();

    // HcclResult code = HcclRecv((void*) requestIdOnDevice, requestIdSize, HCCL_DATA_TYPE_INT8, \
    //     srcRank, g_globalHcclComm, aclStream);
    
    // if (HCCL_SUCCESS != code) {
    //     std::cout << "ERROR HcclSend request id error" << code << std::endl;
    // }
    int deviceId = 0;
    cudaGetDevice(&deviceId);
    auto gpuStream = at::cuda::getCurrentCUDAStream();
    auto cudaStream = gpuStream.stream();

    NCCLCHECK(ncclRecv((void*) requestIdOnDevice, requestIdSize, ncclInt, srcRank, g_globalNcclComm, cudaStream));
    // Synchronize stream
    // CUDACHECK(cudaStreamSynchronize(cudaStream));

    return;
}

void SendBlocksRemote(std::vector<std::pair<at::Tensor, at::Tensor>> srcCaches, \
    std::vector<uint32_t> srcBlocks, uint32_t cacheSize, uint32_t destRank)
{
    int layerNum = srcCaches.size();

    // int deviceId = 0;
    // aclrtGetDevice(&deviceId);
    // auto npuStream = c10_npu:getCurrentNPUStream(deviceId);
    // auto aclStream = npuStream.stream();

    int deviceId = 0;
    cudaGetDevice(&deviceId);
    auto gpuStream = at::cuda::getCurrentCUDAStream();
    auto cudaStream = gpuStream.stream();

    for (int i=0; i < layerNum; i++) {
        at::Tensor srcKeyCache = srcCaches[i].first;
        at::Tensor srcValueCache = srcCaches[i].second;

        for (int j = 0; j < srcBlocks.size(); j++) {
            int blockIdx = srcBlocks[j];
            void *srcKeyCachePtr = srcKeyCache.index({blockIdx}).data_ptr();
            void *srcValueCachePtr = srcValueCache.index({blockIdx}).data_ptr();
            std::cout << "start send key cache: " << srcKeyCachePtr << std::endl;
            if (ncclSuccess != ncclSend(srcKeyCachePtr, cacheSize, ncclInt, destRank,\
                g_globalNcclComm, cudaStream)) {
                std::cout << "[ERROR]  ncclSend key cache error!!" << std::endl;
            }

            std::cout << "start send value cache " << srcValueCachePtr << std::endl;

            if (ncclSuccess != ncclSend(srcValueCachePtr, cacheSize, ncclInt, destRank,\
                g_globalHcclComm, cudaStream)) {
                std::cout << "[ERROR]  ncclSend value cache error!!" << std::endl;
            }
        }
    }
}

void RecvBlocksRemote(std::vector<std::pair<at::Tensor, at::Tensor>> dstCaches, \
    std::vector<uint32_t> dstBlocks, uint32_t cacheSize, uint32_t srcRank)
{
    int layerNum = dstCaches.size();

    // int deviceId = 0;
    // aclrtGetDevice(&deviceId);
    // auto npuStream = c10_npu:getCurrentNPUStream(deviceId);
    // auto aclStream = npuStream.stream();

    int deviceId = 0;
    cudaGetDevice(&deviceId);
    auto gpuStream = at::cuda::getCurrentCUDAStream();
    auto cudaStream = gpuStream.stream();

    for (int i=0; i < layerNum; i++) {
        at::Tensor dstKeyCache = dstCaches[i].first;
        at::Tensor dstValueCache = dstCaches[i].second;

        for (int j = 0; j < dstBlocks.size(); j++) {
            int blockIdx = dstBlocks[j];
            void *dstKeyCachePtr = dstKeyCache.index({blockIdx}).data_ptr();
            void *dstValueCachePtr = dstValueCache.index({blockIdx}).data_ptr();
            std::cout << "start send key cache: " << dstKeyCachePtr << std::endl;
            if (ncclSuccess != ncclRecv(dstKeyCachePtr, cacheSize, ncclInt, srcRank,\
                g_globalHcclComm, cudaStream)) {
                std::cout << "[ERROR]  ncclRecv key cache error!!" << std::endl;
            }

            std::cout << "start send value cache " << dstValueCachePtr << std::endl;

            if (ncclSuccess != ncclRecv(dstValueCachePtr, cacheSize, ncclInt, srcRank,\
                g_globalHcclComm, cudaStream)) {
                std::cout << "[ERROR]  ncclRecv vaule cache error!!" << std::endl;
            }
        }
    }
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
// {
//     m.def("CreateGlobalNcclComm", CreateGlobalNcclComm, "CreateGlobalNcclComm");
//     m.def("SendRequest", &SendRequest, "SendRequest");
//     m.def("RecvRequest", &RecvRequest, "RecvRequest");
//     m.def("SendBlocks", &SendBlocks, "SendBlocks");
//     m.def("RecvBlocks", &RecvBlocks, "RecvBlocks");
// }