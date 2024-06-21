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
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>

const int numGPUs = 2; // Assume two GPUs
#define MAX_NUM_COMM 128
int ready_num = 0;
int running[MAX_NUM_COMM];
std::mutex mtx;
std::condition_variable cv;
bool readyA = false;

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

void nccl_send(int thread_id, long long BUF_SIZE, int NUM_BUFFERS, ncclComm_t comm, int device_id);

void nccl_initA(int thread_id, long long BUF_SIZE, int NUM_BUFFERS, int numGPUs, ncclComm_t comm, ncclUniqueId commId, int device_id){
    // Set device for the current thread
    CUDA_CHECK(cudaSetDevice(device_id));

    // Initialize NCCL communication
    NCCL_CHECK(ncclCommInitRank(&comm, numGPUs, commId, device_id));
    running[thread_id] = 1;
    // std::cout<<"ncclCommInitRank "  << thread_id <<std::endl;

    // std::unique_lock<std::mutex> lock(mtx);
    while(!readyA){
        // cv.wait(lock);
    }
    // std::cout<<"nccl_send  "<<std::endl;
    nccl_send(thread_id, BUF_SIZE, NUM_BUFFERS, comm, device_id);
}

void nccl_send(int thread_id, long long BUF_SIZE, int NUM_BUFFERS, ncclComm_t comm, int device_id) {
    // Set device for the current thread
    // std::cout<<"nccl_send  "<<std::endl;
    CUDA_CHECK(cudaSetDevice(device_id));
    
    // Create streams
    cudaStream_t streams;
    cudaStreamCreate(&streams);

    // Allocate and initialize buffers
    float *send_buf[NUM_BUFFERS];
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        cudaMalloc(&send_buf[i], BUF_SIZE * sizeof(float));
        cudaMemset(send_buf[i], 0, BUF_SIZE * sizeof(float));
    }

    // Warm up: send buffers
    long long buffer_size = BUF_SIZE;
    ncclGroupStart();
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        NCCL_CHECK(ncclSend(send_buf[i], buffer_size, ncclFloat, 1, comm, streams));
    }
    ncclGroupEnd();
    cudaStreamSynchronize(streams);

    // Benchmark: send buffers and measure time
    auto begin = std::chrono::steady_clock::now();
    ncclGroupStart();
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        NCCL_CHECK(ncclSend(send_buf[i], buffer_size, ncclFloat, 1, comm, streams));
    }
    ncclGroupEnd();

    cudaStreamSynchronize(streams);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Thread " << thread_id << " Send Copying time for buffer: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
              << " us" << std::endl;

    // Clean up
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        cudaFree(send_buf[i]);
    }
    cudaStreamDestroy(streams);
    NCCL_CHECK(ncclCommDestroy(comm));
}

int main(int argc, char* argv[]) {
    long long BUF_SIZE = std::atof(argv[1]) * 1000 * 1000;
    int NUM_BUFFERS = std::atoi(argv[2]);
    int NUM_COMM = std::atoi(argv[3]);

    // Initialize array elements to false
    for (int i = 0; i < NUM_COMM; ++i) {
        running[i] = 0;
    }
    std::vector<std::thread> threads;
    // Create NCCL unique IDs and store them in shared memory
    ncclUniqueId commId[NUM_COMM];
    // Initialize NCCL communication
    ncclComm_t comm[NUM_COMM];
    for (int i = 0; i < NUM_COMM; ++i) {
        NCCL_CHECK(ncclGetUniqueId(&commId[i]));
        char filename[256];
        sprintf(filename, "../build/ncclCommId%d", i);
        int fd = open(filename, O_CREAT | O_RDWR, 0666);
        if (fd == -1) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return 1;
        }
        write(fd, &commId[i], sizeof(ncclUniqueId));
        close(fd);
        threads.emplace_back(nccl_initA, i, BUF_SIZE, NUM_BUFFERS, numGPUs, comm[i], commId[i], 0); // device_id set to 0
        while(1){
            if(running[i] == 1){
                std::cout<< " thread i " <<  running[i] << " " << i << std::endl;
                ready_num = ready_num + 1;
                break;
            }
        }
        std::cout<< "for out not " << i << " " << ready_num<< std::endl;
    }
    while(1){
        if(ready_num == NUM_COMM){
            // 主线程准备通知所有线程执行任务
            std::cout << "ready_num " << ready_num << std::endl;
            // std::lock_guard<std::mutex> lock(mtx);
            readyA = true;
            // cv.notify_all();
            break;
        }
    }
    // Join threads
    for (auto& t : threads) {
        t.join();
    }

    return 0;
}
