// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}
#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                      \
    do                                                                     \
    {                                                                      \
        cudaError_t err = stmt;                                            \
        if (err != cudaSuccess)                                            \
        {                                                                  \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                    \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
            return -1;                                                     \
        }                                                                  \
    } while (0)

__global__ void scan(float *array, float *helpArray, int len)
{
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from the host
    __shared__ float T[BLOCK_SIZE * 2];

    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;
    unsigned int id1 = start + 2 * t;
    unsigned int id2 = start + 2 * t + 1;
    int stride = 1;
    int index;

    // copy data into shared memory
    T[2 * t] = (id1 < len) ? array[id1] : 0;
    T[2 * t + 1] = (id2 < len) ? array[id2] : 0;

    // Reduction Step
    while (stride < (BLOCK_SIZE * 2))
    {
        __syncthreads();
        index = (t + 1) * stride * 2 - 1;
        if (index < (BLOCK_SIZE * 2) && (index - stride) >= 0)
            T[index] += T[index - stride];
        stride = stride * 2;
    }

    // Post Scan Step
    stride = BLOCK_SIZE / 2;
    while (stride > 0)
    {
        __syncthreads();
        index = (t + 1) * stride * 2 - 1;
        if ((index + stride) < (BLOCK_SIZE * 2))
            T[index + stride] += T[index];
        stride = stride / 2;
    }

    __syncthreads();

    // copy back to the global memory
    if (id1 < len)
        array[id1] = T[2 * t];
    if (id2 < len)
        array[id2] = T[2 * t + 1];

    // if scan is only partially done
    if ((len > (BLOCK_SIZE * 2)) && (t == 0))
        helpArray[blockIdx.x] = T[(BLOCK_SIZE * 2) - 1];
}

__global__ void scanAdd(float *array, float *helpArray, int len)
{
    unsigned int start = blockIdx.x * blockDim.x;
    unsigned int idx = start + threadIdx.x;

    if ((blockIdx.x > 0) && (len > (BLOCK_SIZE * 2)) && (idx < len))
        array[idx] += helpArray[blockIdx.x - 1];
}

int main(int argc, char **argv)
{
    wbArg_t args;
    float *hostInput;  // The input 1D list
    float *hostOutput; // The output list
    float *deviceArray;
    float *helpArray;
    int numElements; // number of elements in the list
    int numBlocks;

    args = wbArg_read(argc, argv);

    hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float *)malloc(numElements * sizeof(float));

    wbLog(TRACE, "The number of input elements in the input is ",
          numElements);

    numBlocks = numElements / (BLOCK_SIZE << 1);
    if (numBlocks == 0 || numBlocks % (BLOCK_SIZE << 1))
    {
        numBlocks++;
    }

    wbLog(TRACE, "The number of blocks is ",
          numBlocks);

    wbCheck(cudaMalloc((void **)&deviceArray, numElements * sizeof(float)));
    wbCheck(cudaMalloc((void **)&helpArray, numBlocks * sizeof(float)));

    wbCheck(cudaMemcpy(deviceArray, hostInput, numElements * sizeof(float),
                       cudaMemcpyHostToDevice));

    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid(numBlocks, 1, 1);
    dim3 HelpArrayDimGrid(1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    dim3 AddDimBlock(BLOCK_SIZE << 1, 1, 1);

    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    scan<<<DimGrid, DimBlock>>>(deviceArray, helpArray, numElements);
    scan<<<HelpArrayDimGrid, DimBlock>>>(helpArray, NULL, numBlocks);
    scanAdd<<<DimGrid, AddDimBlock>>>(deviceArray, helpArray, numElements);

    cudaDeviceSynchronize();

    wbCheck(cudaMemcpy(hostOutput, deviceArray, numElements * sizeof(float),
                       cudaMemcpyDeviceToHost));

    cudaFree(deviceArray);
    cudaFree(helpArray);

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}