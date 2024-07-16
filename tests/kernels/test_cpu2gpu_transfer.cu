#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
long long DATA_SIZE;
int numstreams;
// 数据拷贝函数
void CopyFromGPUToCPUAsync(const std::vector<void*>& gpuBuffers, const std::vector<void*>& cpuBuffers, cudaStream_t* streams) {
    int j = 0;
    for (size_t i = 0; i < gpuBuffers.size(); ++i) {
        cudaMemcpyAsync(gpuBuffers[i], cpuBuffers[i], DATA_SIZE, cudaMemcpyHostToDevice, streams[j]);
        // cudaMemcpyAsync(cpuBuffers[i], gpuBuffers[i], DATA_SIZE, cudaMemcpyDeviceToHost, streams[j]);
        j = (j + 1) % numstreams;
    }
}

int main(int argc, char* argv[]) {
    // 获取 GPU 设备
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    cudaSetDevice(0);
    // 定义缓存数据大小为 MB
    DATA_SIZE = std::atof(argv[1]) * 1024 * 1024;

    // 定义 GPU 和 CPU 的缓冲区数量
    size_t numBuffers = std::atoi(argv[2]);

    numstreams = std::atoi(argv[3]);

    cudaStream_t streams[numstreams];
    for (int i = 0; i < numstreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // 分配 GPU 内存缓冲区
    std::vector<void*> gpuBuffers(numBuffers);
    for (size_t i = 0; i < numBuffers; ++i) {
        cudaMalloc(&gpuBuffers[i], DATA_SIZE);
    }

    // 分配 CPU 内存缓冲区
    std::vector<void*> cpuBuffers(numBuffers);
    for (size_t i = 0; i < numBuffers; ++i) {
        // cpuBuffers[i] = malloc(DATA_SIZE);
        cudaHostAlloc(&cpuBuffers[i], DATA_SIZE, cudaHostAllocDefault);
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // 进行多次数据拷贝
    CopyFromGPUToCPUAsync(gpuBuffers, cpuBuffers, streams);


    // 等待所有拷贝完成
    for (size_t i = 0; i < numstreams; ++i){
        cudaStreamSynchronize(streams[i]);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Copying time for buffer " << ": " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " us" << std::endl;


    begin = std::chrono::steady_clock::now();

    // 进行多次数据拷贝
    CopyFromGPUToCPUAsync(gpuBuffers, cpuBuffers, streams);

    // 等待所有拷贝完成
    for (size_t i = 0; i < numstreams; ++i){
        cudaStreamSynchronize(streams[i]);
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
        cudaFreeHost(cpuBuffer);
    }

    for (size_t i = 0; i < numstreams; ++i){
        cudaStreamDestroy(streams[i]);
    }
    return 0;
}