#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

#define DIM 10000

__device__ struct cuComplex {
    float r, i;
};

__device__ float cu_magnitude(cuComplex a) {
    return ((a.r * a.r) + (a.i * a.i));
}

__device__ void cu_add(cuComplex a, cuComplex b, cuComplex *res) {
    res->r = a.r + b.r;
    res->i = a.i + b.i;
}

__device__ void cu_mul(cuComplex a, cuComplex b, cuComplex *res) {
    res->r = a.r * b.r - a.i * b.i;
    res->i = a.r * b.r + a.i * b.i;
}

__device__ int cu_julia(int x, int y, cuComplex c) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex a, r1, r2;
    a.r = jx, a.i = jy;

    for (int i = 0; i < 1000; i++) {
        cu_mul(a, a, &r1);
        cu_add(r1, c, &r2);

        if (cu_magnitude(r2) > 4.0f) {
            return 0;
        }
        a.r = r2.r;
        a.i = r2.i;
    }
    return 1;
}

__global__ void cu_kernel(unsigned char *d_image, cuComplex c) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= DIM || y >= DIM) return;

    int offset = (x + y * DIM) * 4;
    int julia_val = cu_julia(x, y, c);

    d_image[offset + 0] = 0;
    d_image[offset + 1] = 0;
    d_image[offset + 2] = 255 * julia_val;
    d_image[offset + 3] = 255;
}

void cu_saveImage(const string &filename, cuComplex c) {
    cv::Mat image(DIM, DIM, CV_8UC4, cv::Scalar(0, 0, 0, 255));
    unsigned char *d_image;

    cudaMalloc((void **)&d_image, DIM * DIM * 4);
    cudaMemset(d_image, 0, DIM * DIM * 4);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((DIM + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (DIM + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cu_kernel<<<numBlocks, threadsPerBlock>>>(d_image, c);
    cudaDeviceSynchronize();

    cudaMemcpy(image.data, d_image, DIM * DIM * 4, cudaMemcpyDeviceToHost);
    cudaFree(d_image);

    cv::imwrite(filename, image);
}

struct Complex {
    float r, i;
};

float magnitude(struct Complex a) {
    return ((a.r * a.r) + (a.i * a.i));
}

void add(struct Complex a, struct Complex b, struct Complex *res) {
    res->r = a.r + b.r;
    res->i = a.i + b.i;
}

void mul(struct Complex a, struct Complex b, struct Complex *res) {
    res->r = a.r * b.r - a.i * b.i;
    res->i = a.r * b.r + a.i * b.i;
}

int julia(int x, int y, struct Complex c) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    struct Complex a, r1, r2;
    a.r = jx, a.i = jy;

    for (int i = 0; i < 1000; i++) {
        mul(a, a, &r1);
        add(r1, c, &r2);

        if (magnitude(r2) > 4.0f) {
            return 0;
        }
        a.r = r2.r;
        a.i = r2.i;
    }
    return 1;
}

void kernelFun(unsigned char *nice, struct Complex c) {
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            int offset = x + y * DIM;
            int julia_val = julia(x, y, c);
            nice[offset * 4 + 0] = 0;
            nice[offset * 4 + 1] = 0;
            nice[offset * 4 + 2] = 255 * julia_val;
            nice[offset * 4 + 3] = 255;
        }
    }
}

void saveImage(const string &filename, struct Complex c) {
    cv::Mat image(DIM, DIM, CV_8UC4, cv::Scalar(0, 0, 0, 255));
    unsigned char *nice = image.data;
    kernelFun(nice, c);
    cv::imwrite(filename, image);
}

int main() {
    Complex c1 = {-0.70176, -0.3842};
    Complex c2 = {0.285, 0.01};
    cuComplex c3 = {-0.70176, -0.3842};
    cuComplex c4 = {0.285, 0.01};

    auto cpu_start = chrono::high_resolution_clock::now();
    saveImage("CPU_connected.png", c1);
    saveImage("CPU_fatou_dust.png", c2);
    auto cpu_end = chrono::high_resolution_clock::now();
    chrono::duration<double> cpu_duration = cpu_end - cpu_start;

    auto gpu_start = chrono::high_resolution_clock::now();
    cu_saveImage("GPU_connected.png", c3);
    cu_saveImage("GPU_fatou_dust.png", c4);
    auto gpu_end = chrono::high_resolution_clock::now();
    chrono::duration<double> gpu_duration = gpu_end - gpu_start;

    cout << "CPU Execution Time: " << cpu_duration.count() << " seconds\n";
    cout << "GPU Execution Time: " << gpu_duration.count() << " seconds\n";
    cout << "Speedup: " << cpu_duration.count() / gpu_duration.count() << "x\n";

    return 0;
}

// To run the Code,
// nvcc -o julia_sets julia_sets.cu pkg-config --cflags --libs opencv4 -diag-suppress=611

// Output:
// CPU Execution Time: 1089.6 seconds
// GPU Execution Time: 2.95337 seconds
// Speedup: 368.934x

// For a popout window.
// cv::namedWindow("Julia Set", cv::WINDOW_NORMAL);
// cv::imshow("Julia Set", image);