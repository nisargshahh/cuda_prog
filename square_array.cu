#include <iostream>
#include <cuda.h>
#include <chrono>

__global__ void squareArrayGPU(float *d_input, float *d_output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_output[idx] = d_input[idx] * d_input[idx];
    }
}

void squareArrayCPU(float *h_input, float *h_output, int n) {
    for (int i = 0; i < n; i++) {
        h_output[i] = h_input[i] * h_input[i];
    }
}

int main() {
    const int N = 1 << 20;
    const int THREADS_PER_BLOCK = 1024;
    const int BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    float *h_input = new float[N];
    float *h_output_cpu = new float[N];
    float *h_output_gpu = new float[N];

    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i);
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    auto gpu_start = std::chrono::high_resolution_clock::now();
    squareArrayGPU<<<BLOCKS, THREADS_PER_BLOCK>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    squareArrayCPU(h_input, h_output_cpu, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i++) {
        if (fabs(h_output_gpu[i] - h_output_cpu[i]) > 1e-5) {
            std::cerr << "Mismatch at index " << i << std::endl;
            break;
        }
    }

    std::chrono::duration<double> gpu_duration = gpu_end - gpu_start;
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;

    std::cout << "GPU Execution Time: " << gpu_duration.count() << " seconds\n";
    std::cout << "CPU Execution Time: " << cpu_duration.count() << " seconds\n";
    std::cout << "Speedup: " << cpu_duration.count() / gpu_duration.count() << "x\n";

    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
