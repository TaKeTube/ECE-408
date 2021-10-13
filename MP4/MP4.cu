#include <wb.h>

#define wbCheck(stmt)                                              \
    do                                                             \
    {                                                              \
        cudaError_t err = stmt;                                    \
        if (err != cudaSuccess)                                    \
        {                                                          \
            wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err)); \
            wbLog(ERROR, "Failed to run stmt ", #stmt);            \
            return -1;                                             \
        }                                                          \
    } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define MASK_RADIUS 1
#define TILE_WIDTH 8
#define BLOCK_WIDTH (TILE_WIDTH + MASK_WIDTH - 1)

//@@ Define constant memory for device kernel here
__constant__ float M[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size)
{
    //@@ Insert kernel code here
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int x_o = blockIdx.x * TILE_WIDTH + tx;
    int y_o = blockIdx.y * TILE_WIDTH + ty;
    int z_o = blockIdx.z * TILE_WIDTH + tz;

    int x_i = x_o - MASK_RADIUS;
    int y_i = y_o - MASK_RADIUS;
    int z_i = z_o - MASK_RADIUS;

    __shared__ float input_ds[BLOCK_WIDTH][BLOCK_WIDTH][BLOCK_WIDTH];

    // copy data from global memory to shared memory
    if ((x_i >= 0) && (x_i < x_size) &&
        (y_i >= 0) && (y_i < y_size) &&
        (z_i >= 0) && (z_i < z_size))
    {
        input_ds[tz][ty][tx] = input[z_i * (y_size * x_size) + y_i * (x_size) + x_i];
    }
    else
    {
        input_ds[tz][ty][tx] = 0.0f;
    }
    __syncthreads();

    if (tx < TILE_WIDTH && ty < TILE_WIDTH && tz < TILE_WIDTH && x_o < x_size && y_o < y_size && z_o < z_size)
    {
        float Pvalue = 0;
        for (int i = 0; i < MASK_WIDTH; i++)
            for (int j = 0; j < MASK_WIDTH; j++)
                for (int k = 0; k < MASK_WIDTH; k++)
                    Pvalue += M[i][j][k] * input_ds[tz + i][ty + j][tx + k];

        output[z_o * (y_size * x_size) + y_o * (x_size) + x_o] = Pvalue;
    }
}

int main(int argc, char *argv[])
{
    wbArg_t args;
    int z_size;
    int y_size;
    int x_size;
    int inputLength, kernelLength, tensorLength;
    float *hostInput;
    float *hostKernel;
    float *hostOutput;
    float *deviceInput;
    float *deviceOutput;

    args = wbArg_read(argc, argv);

    // Import data
    hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostKernel =
        (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
    hostOutput = (float *)malloc(inputLength * sizeof(float));

    // First three elements are the input dimensions
    z_size = hostInput[0];
    y_size = hostInput[1];
    x_size = hostInput[2];
    wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
    assert(z_size * y_size * x_size == inputLength - 3);
    assert(kernelLength == 27);
    tensorLength = inputLength - 3;

    //@@ Allocate GPU memory here
    // Recall that inputLength is 3 elements longer than the input data
    // because the first  three elements were the dimensions
    cudaMalloc((void **)&deviceInput, tensorLength * sizeof(float));
    cudaMalloc((void **)&deviceOutput, tensorLength * sizeof(float));

    //@@ Copy input and kernel to GPU here
    // Recall that the first three elements of hostInput are dimensions and
    // do
    // not need to be copied to the gpu
    // copy input data
    cudaMemcpy(deviceInput, &hostInput[3], tensorLength * sizeof(float), cudaMemcpyHostToDevice);
    // copy kernel data
    cudaMemcpyToSymbol(M, hostKernel, MASK_WIDTH * MASK_WIDTH * MASK_WIDTH * sizeof(float));

    //@@ Initialize grid and block dimensions here
    dim3 DimGrid(ceil(((float)x_size) / TILE_WIDTH), ceil(((float)y_size) / TILE_WIDTH), ceil(((float)z_size) / TILE_WIDTH));
    dim3 DimBlock(BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH);

    //@@ Launch the GPU kernel here
    conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

    cudaDeviceSynchronize();

    //@@ Copy the device memory back to the host here
    // Recall that the first three elements of the output are the dimensions
    // and should not be set here (they are set below)
    cudaMemcpy(&hostOutput[3], deviceOutput, tensorLength * sizeof(float), cudaMemcpyDeviceToHost);

    // Set the output dimensions for correctness checking
    hostOutput[0] = z_size;
    hostOutput[1] = y_size;
    hostOutput[2] = x_size;
    wbSolution(args, hostOutput, inputLength);

    // for(int i = 0; i < 3; ++i)
    //  for(int j = 0; j < 3; ++j)
    //    for(int k = 0; k < 3; ++k)
    //      wbLog(TRACE, " ", i, " ", j, " ", k, " ", hostOutput[3 + i * (y_size * x_size) + j * (x_size) + k]);

    // Free device memory
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    // Free host memory
    free(hostInput);
    free(hostOutput);
    return 0;
}
