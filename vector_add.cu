#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

__global__ void gpuAdd(float *d_A, float *d_B, float * d_C, int n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n){
        d_C[i] = d_A[i] + d_B[i];
    }
}

vector<float> cpuAdd(vector<float> h_A, vector<float> h_B, int n){
    vector<float> res(n, 0);
    for (int i = 0; i < n; i++){
        res[i] = h_A[i] + h_B[i];
    }
    return res;
}

int main(){
    int n;
    cin>>n;

    vector<float> h_A(n,0);
    vector<float> h_B(n,0);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> pdid(0, 1000000000);

    for (int t = 0; t < n; t++){
        h_A[t] = pdid(gen);
        h_B[t] = pdid(gen);
    }

    vector<float> h_C (n, 0);
    const int threads = 1024;
    const int blocks = (n + threads - 1) / threads;

    auto cpu_start = std::chrono::high_resolution_clock::now();
    vector<float> fin = cpuAdd(h_A, h_B, n);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**) &d_A, n * sizeof(float));
    cudaMalloc((void**) &d_B, n * sizeof(float));
    cudaMalloc((void**) &d_C, n * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), n * sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), n * sizeof(float) , cudaMemcpyHostToDevice);
    // cudaMemcpy(d_C, h_C.data(), n * sizeof(float) , cudaMemcpyHostToDevice);

    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpuAdd<<<blocks, threads>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();
    auto gpu_end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_C.data(), d_C, n * sizeof(float) , cudaMemcpyDeviceToHost);

    bool bruh = true;
    for (int i = 0; i < n; i++) {
        if (fin[i] != h_C[i]) bruh = false;
    }

    if (bruh) {
        cout<<"GPU is working fine.."<<endl;
        std::chrono::duration<double> gpu_duration = gpu_end - gpu_start;
        std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;

        std::cout << "GPU Execution Time: " << gpu_duration.count() << " seconds\n";
        std::cout << "CPU Execution Time: " << cpu_duration.count() << " seconds\n";
        std::cout << "Speedup: " << cpu_duration.count() / gpu_duration.count() << "x\n";
    } else cout<<"GPU is incorrect."<<endl;
    cout<<endl;
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}