#include <wb.h>

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

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns)
{
    //@@ Insert code to implement matrix multiplication here
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < numCRows) && (Col < numCColumns))
    {
        float sum = 0;
        for (int k = 0; k < numAColumns; ++k)
            sum += A[Row * numAColumns + k] * B[k * numBColumns + Col];
        C[Row * numCColumns + Col] = sum;
    }
}

int main(int argc, char **argv)
{
    wbArg_t args;
    float *hostA; // The A matrix
    float *hostB; // The B matrix
    float *hostC; // The output C matrix
    float *deviceA;
    float *deviceB;
    float *deviceC;
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;    // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set
                     // this)

    args = wbArg_read(argc, argv);

    hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                              &numAColumns);
    hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                              &numBColumns);

    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;

    //@@ Allocate the hostC matrix
    hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));

    //@@ Allocate GPU memory here
    cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(float));
    cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(float));
    cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(float));

    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

    //@@ Initialize the grid and block dimensions here
    int tile_width = 32;
    dim3 DimGrid(ceil(((float)numCColumns) / tile_width), ceil(((float)numCRows) / tile_width), 1);
    dim3 DimBlock(tile_width, tile_width, 1);

    //@@ Launch the GPU Kernel here
    matrixMultiply<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC,
                                          numARows, numAColumns,
                                          numBRows, numBColumns,
                                          numCRows, numCColumns);

    cudaDeviceSynchronize();

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);

    wbLog(TRACE, " Gridx ", ceil(((float)numCColumns) / tile_width), " Gridy ", ceil(((float)numCRows) / tile_width));

    // for(int i = 0; i < numCColumns; ++i)
    //  for(int j = 0; j < numCRows; ++j)
    //    wbLog(TRACE, " ", i, " ", j, " ", hostC[i+j*numCColumns]);

    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
