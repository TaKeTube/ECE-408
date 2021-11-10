// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH    256
#define TILE_WIDTH          32

//@@ insert code here
__global__ 
void FloatToUchar(unsigned char* ucharImage, float* floatImage, int height, int width, int imageChannels){
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    int Channel = threadIdx.z;
    if(Col < width && Row < height) {
        int offset = (Row * width + Col) * imageChannels + Channel;
        ucharImage[offset] = (unsigned char) (255 * floatImage[offset]);
    }
}

__global__ 
void RGBToGray(unsigned char* grayImage, unsigned char* rgbImage, int height, int width, int imageChannels){
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    if(Col < width && Row < height) {
        int grayOffset = Row * width + Col;
        int rgbOffset = grayOffset*imageChannels;
        unsigned char r = rgbImage[rgbOffset];
        unsigned char g = rgbImage[rgbOffset + 1];
        unsigned char b = rgbImage[rgbOffset + 2];
        grayImage[grayOffset] = (unsigned char) (0.21f*r + 0.71f*g + 0.07f*b);
    }
}

__global__ 
void ImageHisto(unsigned char* image, int height, int width, unsigned int* histo){
    __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];

    int linearIdx = threadIdx.x + threadIdx.y * blockDim.x;
    if(linearIdx < HISTOGRAM_LENGTH)
        histo_private[linearIdx] = 0;

    __syncthreads();

    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    if(Col < width && Row < height) {
        int offset = Row * width + Col;
        atomicAdd(&(histo_private[image[offset]]), 1);
    }

    __syncthreads();

    if(linearIdx < HISTOGRAM_LENGTH)
        atomicAdd(&(histo[linearIdx]), histo_private[linearIdx]);
}

__global__ 
void HistCDF(float *cdf, unsigned int* histo, int imgSize){
    __shared__ float T[HISTOGRAM_LENGTH];

    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;
    unsigned int id1 = start + 2 * t;
    unsigned int id2 = start + 2 * t + 1;
    int stride = 1;
    int index;

    // copy histo into shared memory
    T[2 * t] = (float) histo[id1] / imgSize;
    T[2 * t + 1] = (float) histo[id2] / imgSize;

    // Reduction Step
    while (stride < HISTOGRAM_LENGTH)
    {
        __syncthreads();
        index = (t + 1) * stride * 2 - 1;
        if (index < HISTOGRAM_LENGTH && (index - stride) >= 0)
            T[index] += T[index - stride];
        stride = stride * 2;
    }

    // Post Scan Step
    stride = HISTOGRAM_LENGTH / 4;
    while (stride > 0)
    {
        __syncthreads();
        index = (t + 1) * stride * 2 - 1;
        if ((index + stride) < HISTOGRAM_LENGTH)
            T[index + stride] += T[index];
        stride = stride / 2;
    }

    __syncthreads();

    // copy back to the cdf in global memory
    cdf[id1] = T[2 * t];
    cdf[id2] = T[2 * t + 1];
}

__global__
void HistoEqualization(unsigned char* ucharImage, float* cdf, int height, int width, int imageChannels){
    __shared__ float cdfmin;

    if(threadIdx.x == 0 && threadIdx.y == 0)
        cdfmin = cdf[0];

    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    int Channel = threadIdx.z;

    if(Col < width && Row < height) {
        int offset = (Row * width + Col) * imageChannels + Channel;
        ucharImage[offset] = min(max(255*(cdf[ucharImage[offset]] - cdfmin)/(1.0 - cdfmin), 0.0f), 255.0f);
    }
}

__global__ 
void UcharToFloat(unsigned char* ucharImage, float* outputImage, int height, int width, int imageChannels){
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    int Channel = threadIdx.z;
    if(Col < width && Row < height) {
        int offset = (Row * width + Col) * imageChannels + Channel;
        outputImage[offset] = (float) (ucharImage[offset] / 255.0);
    }
}

int main(int argc, char **argv)
{
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float *hostInputImageData;
    float *hostOutputImageData;
    const char *inputImageFile;

    //@@ Insert more code here
    float *deviceInputImage;
    unsigned char *deviceUcharImage;
    unsigned char *deviceGrayImage;
    unsigned int *deviceHisto;
    float* deviceCDF;
    float* deviceOutputImage;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    //@@ insert code here
    cudaMalloc((void **)&deviceInputImage, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **)&deviceUcharImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char));
    cudaMalloc((void **)&deviceGrayImage, imageWidth * imageHeight * sizeof(unsigned char));
    cudaMalloc((void **)&deviceHisto, HISTOGRAM_LENGTH * sizeof(unsigned int));
    cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(float));
    cudaMalloc((void **)&deviceOutputImage, imageWidth * imageHeight * imageChannels * sizeof(float));

    cudaMemcpy(deviceInputImage, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float),
                       cudaMemcpyHostToDevice);

    dim3 DimConvertBlock(TILE_WIDTH, TILE_WIDTH, imageChannels);
    dim3 DimConvertGrid(ceil(((float)imageWidth)/TILE_WIDTH), ceil(((float)imageHeight)/TILE_WIDTH), 1);
    dim3 DimHistCDFBlock(HISTOGRAM_LENGTH/2, 1, 1);
    dim3 DimHistCDFGrid(1, 0, 0);

    FloatToUchar<<<DimConvertGrid, DimConvertBlock>>>(deviceUcharImage, deviceInputImage, imageHeight, imageWidth, imageChannels);
    RGBToGray<<<DimConvertGrid, DimConvertBlock>>>(deviceGrayImage, deviceUcharImage, imageHeight, imageWidth, imageChannels);
    ImageHisto<<<DimConvertGrid, DimConvertBlock>>>(deviceGrayImage, imageHeight, imageWidth, deviceHisto);
    HistCDF<<<DimHistCDFGrid, DimHistCDFBlock>>>(deviceCDF, deviceHisto, imageHeight*imageWidth);
    HistoEqualization<<<DimConvertGrid, DimConvertBlock>>>(deviceUcharImage, deviceCDF, imageHeight, imageWidth, imageChannels);
    UcharToFloat<<<DimConvertGrid, DimConvertBlock>>>(deviceUcharImage, deviceOutputImage, imageHeight, imageWidth, imageChannels);

    cudaDeviceSynchronize();

    cudaMemcpy(hostOutputImageData, deviceOutputImage, imageWidth * imageHeight * imageChannels * sizeof(float),
                       cudaMemcpyDeviceToHost);

    cudaFree(deviceInputImage);
    cudaFree(deviceUcharImage);
    cudaFree(deviceGrayImage);
    cudaFree(deviceHisto);
    cudaFree(deviceCDF);
    cudaFree(deviceOutputImage);

    wbSolution(args, outputImage);

    //@@ insert code here
    return 0;
}
