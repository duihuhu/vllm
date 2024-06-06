#include "gpu_ops.h"

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
ncclComm_t comm[4];
ncclUniqueId commId[4];

constexpr int MAX_SHM_NAME_LENGTH = 256;
long long index_time = 0;
long long send_time = 0;
long long nccl_time = 0;
long long nccl_num = 0;
int num_comm = 0;
int32_t CreateInternalNcclComm(int32_t rank, int32_t NumDevice, ncclComm_t& comm, ncclUniqueId uniqueId) {
    int32_t g_tpSize = NumDevice;
    if (NumDevice <=1) {
        g_tpSize = 1;
        return 0;
    }
    std::cout << "Start init CreateInternalNcclComm  NCCL Comm Success" << std::endl;
    NCCLCHECK(ncclCommInitRank(&comm, NumDevice, uniqueId ,rank));
    std::cout << "Create CreateInternalNcclComm NCCL Comm Success" << std::endl;
    return 0;
}

int32_t CreateGlobalMulNcclComm(int32_t rank, int32_t NumDevice , int32_t num_comms) {
    constexpr int32_t ROOT_RANK = 0;
    constexpr int32_t TIME_OUT = 180;
    int32_t g_tpSize = NumDevice;
    if (NumDevice <=1) {
        g_tpSize = 1;
        return 0;
    }
    if (rank == ROOT_RANK) {
        for (int i = 0; i < num_comms; ++i) {
            int shm_fd;
            int shmSize = sizeof(ncclUniqueId);
            ncclGetUniqueId(&commId[i]);
            char shmName[MAX_SHM_NAME_LENGTH];
            sprintf(shmName, "NcclRootInfo%d", i);
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
            ncclGetUniqueId(&commId[i]);
            memcpy(shmaddr, &commId[i], shmSize);
            // 解除映射
            if (munmap(shmaddr, shmSize) == -1) {
                perror("munmap");
                exit(1);
            }
            ncclCommInitRank(&comm[i], NumDevice, commId[i], rank);
            std::cout<<"AAA ncclCommInitRank "<<std::endl;
        }

    } else {
        for (int i = 0; i < num_comms; ++i) {
            int shm_fd;
            int shmSize = sizeof(ncclUniqueId);
            char shmName[MAX_SHM_NAME_LENGTH];
            sprintf(shmName, "NcclRootInfo%d", i);
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
            memcpy(&commId[i], shmaddr, shmSize);
            // 解除映射
            if (munmap(shmaddr, shmSize) == -1) {
                perror("munmap");
                exit(1);
            }

            ncclCommInitRank(&comm[i], NumDevice, commId[i] ,rank);
            sleep(1);
            std::cout<<"BBB ncclCommInitRank "<<std::endl;
        }
    }
    return 0;
}


int32_t CreateGlobalNcclComm(int32_t rank, int32_t NumDevice , int32_t num_comms) {
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
            if (ncclSuccess != ncclSend(srcKeyCachePtr, cacheSize, ncclInt, destRank,\
                comm[num_comm], cudaStream)) {
                std::cout << "[ERROR]  ncclSend key cache error!!" << std::endl;
            }
            if (ncclSuccess != ncclSend(srcValueCachePtr, cacheSize, ncclInt, destRank,\
                comm[num_comm], cudaStream)) {
                std::cout << "[ERROR]  ncclSend value cache error!!" << std::endl;
            }
            num_comm = (num_comm + 1) % 4;
        }
    }
    NCCLCHECK(ncclGroupEnd());
}

void RecvBlocksRemote(std::vector<std::pair<at::Tensor, at::Tensor>> dstCaches, \
    std::vector<uint32_t> dstBlocks, uint32_t cacheSize, uint32_t srcRank)
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
            if (ncclSuccess != ncclRecv(dstKeyCachePtr, cacheSize, ncclInt, srcRank,\
                comm[num_comm], cudaStream)) {
                std::cout << "[ERROR]  ncclRecv key cache error!!" << std::endl;
            }
            if (ncclSuccess != ncclRecv(dstValueCachePtr, cacheSize, ncclInt, srcRank,\
                comm[num_comm], cudaStream)) {
                std::cout << "[ERROR]  ncclRecv vaule cache error!!" << std::endl;
            }
            num_comm = (num_comm + 1) % 4;
        }
    }
    NCCLCHECK(ncclGroupEnd());
}

void SendLayerBlocks(std::vector<std::pair<at::Tensor, at::Tensor>> srcCaches, \
    std::vector<uint32_t> srcBlocks, uint32_t cacheSize, uint32_t destRank, uint32_t layer)
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
            g_globalNcclComm, cudaStream)) {
            std::cout << "[ERROR]  ncclSend key cache error!!" << std::endl;
        }
        if (ncclSuccess != ncclSend(srcValueCachePtr, cacheSize, ncclInt, destRank,\
            g_globalNcclComm, cudaStream)) {
            std::cout << "[ERROR]  ncclSend value cache error!!" << std::endl;
        }
    }
    NCCLCHECK(ncclGroupEnd());
}

void HandleNcclCommDestroy()
{
    ncclCommDestroy(g_globalNcclComm);
}
