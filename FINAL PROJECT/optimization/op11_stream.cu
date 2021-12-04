#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 8

__global__ void conv_forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int m = blockIdx.x;
    int b = blockIdx.z;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    float acc = 0.0f;
    if((h < H_out) && (w < W_out)){
        for(int c = 0; c < C; c++){
            for(int p = 0; p < K; p++)
                for(int q = 0; q < K; q++)
                        acc += x4d(b, c, h + p, w + q) * k4d(m, c, p, q);
        }
        y4d(b, m, h, w) = acc;
    }

#undef y4d
#undef x4d
#undef k4d
}


__host__ void GPUInterface::conv_forward_gpu_prolog(float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
#define STREAM_NUM 8

    // Allocate memory and copy over the relevant data structures to the GPU
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int x_batch_size = (B * C * H * W) / STREAM_NUM;
    int y_batch_size = (B * M * H_out * W_out) / STREAM_NUM;

    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int Y = W_grid * H_grid;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(M, Y, B);

    cudaMalloc((void**)device_y_ptr, B * M * H_out * W_out * sizeof(float));
    cudaMalloc((void**)device_x_ptr, B * C * H * W * sizeof(float));
    cudaMalloc((void**)device_k_ptr, M * C * K * K * sizeof(float));

    cudaStream_t stream[STREAM_NUM];
    for (int i = 0; i < STREAM_NUM; i++)
        cudaStreamCreate(&stream[i]);

    cudaMemcpyAsync(*device_k_ptr, host_k, M * C * K * K * sizeof(float), cudaMemcpyHostToDevice, stream[0]);
    for (int i = 0; i < STREAM_NUM; i++){
        int x_offset = x_batch_size * i;
        int y_offset = y_batch_size * i;
        cudaMemcpyAsync((*device_x_ptr) + x_offset, host_x + x_offset, x_batch_size * sizeof(float), cudaMemcpyHostToDevice, stream[i]);
        conv_forward_kernel<<<gridDim, blockDim, 0, stream[i]>>>((*device_y_ptr) + y_offset, (*device_x_ptr) + x_offset, *device_k_ptr, B, M, C, H, W, K);
        cudaMemcpyAsync(host_y + y_offset, (*device_y_ptr) + y_offset, y_batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < STREAM_NUM; i++)
        cudaStreamDestroy(stream[i]);

    // Free device memory
    cudaFree(device_x_ptr);
    cudaFree(device_y_ptr);
    cudaFree(device_k_ptr);

#undef STREAM_NUM
}

__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    return;
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    return;
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
