#include <stdio.h>
#include <stdlib.h>
#include <nccl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define CUDA_CHECK(cmd) do { \
    cudaError_t e = cmd; \
    if (e != cudaSuccess) { \
        printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define NCCL_CHECK(cmd) do { \
    ncclResult_t r = cmd; \
    if (r != ncclSuccess) { \
        printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

int main(int argc, char* argv[]) {
    long long BUF_SIZE =  std::atof(argv[1]) * 1000 * 1000;
    int NUM_BUFFERS = std::atoi(argv[2]);
    int NUM_STREAM = std::atoi(argv[3]);
    const int numGPUs = 2;
    cudaStream_t streams[NUM_STREAM];
    // Allocate GPU buffers
    float *send_buf[NUM_BUFFERS];
    // Initialize CUDA and NCCL
    CUDA_CHECK(cudaSetDevice(0));
    for (int i = 0; i < NUM_STREAM; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        cudaMalloc(&send_buf[i], BUF_SIZE * sizeof(float));
        cudaMemset(send_buf[i], 0, BUF_SIZE * sizeof(float));
    }
    ncclUniqueId commId[NUM_STREAM];
    ncclComm_t comm[NUM_STREAM];
    for (int i = 0; i < NUM_STREAM; ++i) {
        NCCL_CHECK(ncclGetUniqueId(&commId[i]));
        // 将唯一标识符写入共享内存
        char filename[256];
        sprintf(filename, "/tmp/ncclCommId%d", i);

        int fd = open(filename, O_CREAT | O_RDWR, 0666);
        if (fd == -1) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return 1;
        }
        write(fd, &commId[i], sizeof(ncclUniqueId));
        close(fd);
        NCCL_CHECK(ncclCommInitRank(&comm[i], numGPUs, commId[i], 0));
        std::cout<<"AAA"<<std::endl;
    }
    std::cout<<"AAA end "<<std::endl;
    // ncclGroupStart();
    long long buffer_size = BUF_SIZE ;
    int count = 0;
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        ncclResult_t send_result = ncclSend(send_buf[i], buffer_size, ncclFloat, 1, comm[count], streams[count]);
        if (send_result != ncclSuccess) {
            std::cout<<"send error"<<std::endl;
        }
        count = count + 1;
        count = count % NUM_STREAM;
    }
    // ncclGroupEnd();
    for (int i = 0; i < NUM_STREAM; ++i) {
        cudaStreamSynchronize(streams[i]);
    }
    ncclGroupStart();
    count = 0;
    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        ncclResult_t send_result = ncclSend(send_buf[i], buffer_size, ncclFloat, 1, comm[count], streams[count]);
        if (send_result != ncclSuccess) {
            std::cout<<"send error"<<std::endl;
        }
        count = count + 1;
        count = count % NUM_STREAM;
    }

    ncclGroupEnd();
    for (int i = 0; i < NUM_STREAM; ++i) {
        cudaStreamSynchronize(streams[i]);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Send Copying time for buffer " << ": " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " us" << std::endl;

    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // std::cout << "Send Copying time for buffer " << ": " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " us" << std::endl;


    std::cout<<"send success"<<std::endl;
    // Finalize CUDA and NCCL
   for (int i = 0; i < NUM_BUFFERS; ++i) {
        cudaFree(send_buf[i]);
    }
    for (int i = 0; i < NUM_STREAM; ++i) {
        NCCL_CHECK(ncclCommDestroy(comm[i]));
    }
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
