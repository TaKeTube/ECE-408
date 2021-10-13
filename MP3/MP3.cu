
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

#define TILE_WIDTH 32

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
    __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    int mnum = (numAColumns - 1) / TILE_WIDTH + 1;
    float sum = 0;

    for (int m = 0; m < mnum; ++m)
    {
        if ((Row < numARows) && (m * TILE_WIDTH + tx < numAColumns))
        {
            subTileA[ty][tx] = A[Row * numAColumns + m * TILE_WIDTH + tx];
        }
        else
        {
            subTileA[ty][tx] = 0;
        }
        if ((Col < numBColumns) && (m * TILE_WIDTH + ty < numBRows))
        {
            subTileB[ty][tx] = B[(m * TILE_WIDTH + ty) * numBColumns + Col];
        }
        else
        {
            subTileB[ty][tx] = 0;
        }
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += subTileA[ty][k] * subTileB[k][tx];
        __syncthreads();
    }

    if ((Row < numCRows) && (Col < numCColumns))
        C[Row * numCColumns + Col] = sum;
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
    dim3 DimGrid(ceil(((float)numCColumns) / TILE_WIDTH), ceil(((float)numCRows) / TILE_WIDTH), 1);
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    //@@ Launch the GPU Kernel here
    matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC,
                                                numARows, numAColumns,
                                                numBRows, numBColumns,
                                                numCRows, numCColumns);

    cudaDeviceSynchronize();

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);

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
