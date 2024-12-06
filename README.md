# CUDA Programming with C++
CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA. It allows developers to harness the computing power of NVIDIA GPUs for general-purpose computing tasks, often referred to as GPGPU (General-Purpose Graphics Processing Unit). CUDA enables software developers to use the C, C++, and Fortran programming languages to write highly parallel code that can execute on GPUs, achieving significant performance improvements over traditional CPU-based execution.

In this guide, we will explore CUDA programming with C++, focusing on key concepts, tools, and techniques to maximize the performance of your applications using GPUs. CUDA's ability to perform computations in parallel using thousands of GPU threads allows developers to efficiently solve problems that require large amounts of processing power, such as scientific simulations, machine learning, and image processing.

## CUDA Programming Model
CUDA programming involves two primary components:
1. **Host Code (CPU-side):** The host code runs on the CPU and is responsible for managing memory, handling I/O operations, and launching GPU kernels.
2. **Device Code (GPU-side):** The device code runs on the GPU and performs the actual computation in parallel across many threads.

In CUDA, the term **kernel** refers to a function that executes on the GPU. Kernels are launched by the host and run in parallel on a large number of threads. The parallel execution model is crucial for accelerating compute-intensive tasks, as many operations can be performed simultaneously.

### CUDA Kernels
A **kernel** is a special function written using CUDA syntax that is executed on the GPU. It is marked with the `__global__` keyword and can be called from the host code. CUDA kernels are launched by specifying the number of threads and the number of blocks in a grid structure. The `threadIdx`, `blockIdx`, and `blockDim` variables are used to identify the position of each thread within a block and grid.

There are a lot of examples in this repo.
