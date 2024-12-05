#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

__global__ void matMulKernel(const int* A, const int* B, int* C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        int sum = 0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

vector<vector<int>> cpuMatMul(vector<vector<int>> A, vector<vector<int>> B) {
    int m = A.size(), n1 = A[0].size(), n2 = B.size(), p = B[0].size();
    vector<vector<int>> C(m, vector<int>(p, 0));
    if (n1 != n2) {
        cout << "Incorrect Matrix Multiplication.";
        return C;
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n1; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

int main() {
    int m, n1, n2, p;
    cin >> m >> n1 >> n2 >> p;

    if (n1 != n2) {
        cout << "Matrix dimensions are incompatible for multiplication.\n";
        return 0;
    }

    mt19937 mt(random_device{}());
    uniform_int_distribution<int> dist(1, 10000);

    vector<vector<int>> A(m, vector<int>(n1)), B(n2, vector<int>(p));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n1; j++) {
            A[i][j] = dist(mt);
        }
    }
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < p; j++) {
            B[i][j] = dist(mt);
        }
    }

    vector<int> flatA(m * n1), flatB(n2 * p), flatC(m * p, 0);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n1; j++) {
            flatA[i * n1 + j] = A[i][j];
        }
    }
    for (int i = 0; i < n2; i++) {
        for (int j = 0; j < p; j++) {
            flatB[i * p + j] = B[i][j];
        }
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * n1 * sizeof(int));
    cudaMalloc(&d_B, n2 * p * sizeof(int));
    cudaMalloc(&d_C, m * p * sizeof(int));

    cudaMemcpy(d_A, flatA.data(), m * n1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, flatB.data(), n2 * p * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((p + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    auto gpu_start = chrono::high_resolution_clock::now();
    matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n1, p);
    cudaDeviceSynchronize();
    auto gpu_end = chrono::high_resolution_clock::now();

    cudaMemcpy(flatC.data(), d_C, m * p * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    vector<vector<int>> C2(m, vector<int>(p));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C2[i][j] = flatC[i * p + j];
        }
    }

    auto cpu_start = chrono::high_resolution_clock::now();
    vector<vector<int>> C1 = cpuMatMul(A, B);
    auto cpu_end = chrono::high_resolution_clock::now();

    chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    chrono::duration<double> gpu_duration = gpu_end - gpu_start;

    cout << "CPU Execution Time: " << cpu_duration.count() << " seconds\n";
    cout << "GPU Execution Time: " << gpu_duration.count() << " seconds\n";
    cout << "Speedup: " << cpu_duration.count() / gpu_duration.count() << "x\n";

    return 0;
}
