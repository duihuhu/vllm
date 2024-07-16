#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

// 定义数据大小为 25MB
long long DATA_SIZE = 6.25 * 1024 * 1024;


// 数据拷贝函数
void CopyFromCPUToGPUAsync(const std::vector<void*>& gpuBuffers, const std::vector<void*>& cpuBuffers, int buffers, cudaStream_t streams) {
    for (size_t i = 0; i < gpuBuffers.size(); ++i) {
        cudaMemcpyAsync(gpuBuffers[i], cpuBuffers[i], buffers, cudaMemcpyHostToDevice, streams);
    }
}

int main() {
    // 获取 GPU 设备
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);


    // 定义 GPU 和 CPU 的缓冲区数量
    size_t numBuffers = 1; 

    cudaStream_t streams;
    for (int i = 0; i < numBuffers; ++i) {
        cudaStreamCreate(&streams);
    }


    size_t buffers = DATA_SIZE / numBuffers;
    // 分配 GPU 内存缓冲区
    std::vector<void*> gpuBuffers(numBuffers);
    for (size_t i = 0; i < numBuffers; ++i) {
        cudaMalloc(&gpuBuffers[i], buffers);
    }

    // 分配 CPU 内存缓冲区
    std::vector<void*> cpuBuffers(numBuffers);
    for (size_t i = 0; i < numBuffers; ++i) {
        cpuBuffers[i] = calloc(buffers, 1);
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // 进行多次数据拷贝
    for (size_t i = 0; i < numBuffers; ++i) {
        CopyFromCPUToGPUAsync(gpuBuffers, cpuBuffers, buffers, streams);
    }

    for (int i = 0; i < numBuffers; ++i) {
        // 等待所有拷贝完成
        cudaStreamSynchronize(streams);
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Copying time for buffer " << ": " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " us" << std::endl;


    begin = std::chrono::steady_clock::now();

    // 进行多次数据拷贝
    for (size_t i = 0; i < numBuffers; ++i) {
        CopyFromCPUToGPUAsync(gpuBuffers, cpuBuffers, buffers, streams);
    }

    for (int i = 0; i < numBuffers; ++i) {
        // 等待所有拷贝完成
        cudaStreamSynchronize(streams);
    }
    end = std::chrono::steady_clock::now();
    std::cout << "Copying time for buffer " << ": " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " us" << std::endl;

    begin = std::chrono::steady_clock::now();

    // 进行多次数据拷贝
    for (size_t i = 0; i < numBuffers; ++i) {
       CopyFromCPUToGPUAsync(gpuBuffers, cpuBuffers, buffers, streams);
    }

    for (int i = 0; i < numBuffers; ++i) {
        // 等待所有拷贝完成
        cudaStreamSynchronize(streams);
    }
    end = std::chrono::steady_clock::now();
    std::cout << "Copying time for buffer " << ": " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " us" << std::endl;

    // 检查拷贝是否成功
    for (size_t i = 0; i < numBuffers; ++i) {
        if (cudaGetLastError() != cudaSuccess) {
            std::cerr << "CUDA error in buffer " << i << ": " << cudaGetLastError() << std::endl;
            return 1;
        }
    }

    std::cout << "All data copied from GPU to CPU successfully." << std::endl;

    // 释放 GPU 和 CPU 内存
    for (void* gpuBuffer : gpuBuffers) {
        cudaFree(gpuBuffer);
    }
    for (void* cpuBuffer : cpuBuffers) {
        free(cpuBuffer);
    }

    // for (int i = 0; i < 40; ++i) {
    cudaStreamDestroy(streams);
    // }

    return 0;
}