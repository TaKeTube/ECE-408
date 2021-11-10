// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH    256
#define RGB_CHANNELS        3
#define TILE_SIZE           32

//@@ insert code here
__global__ 
void FloatToUchar(unsigned char* ucharImage, float* floatImage, int height, int weight){
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    int Channel = threadIdx.z;
    if(Col < width && Row < height) {
        int offset = (Row * width + Col) * RGB_CHANNELS + Channel;
        ucharImage[offset] = (unsigned char) (255 * floatImage[offset]);
    }
}

__global__ 
void RGBToGray(unsigned char* grayImage, unsigned char* rgbImage, int height, int weight){
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    if(Col < width && Row < height) {
        int grayOffset = Row * width + Col;
        int rgbOffset = grayOffset*RGB_CHANNELS;
        unsigned char r = rgbImage[rgbOffset];
        unsigned char g = rgbImage[rgbOffset + 1];
        unsigned char b = rgbImage[rgbOffset + 2];
        grayImage[grayOffset] = (unsigned char) (0.21f*r + 0.71f*g + 0.07f*b)
    }
}

__global__ 
void ImageHisto(unsigned char* image, int height, int weight, unsigned char* histo){
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
void HistoCDF(unsigned char* histo){
    
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

    wbSolution(args, outputImage);

    //@@ insert code here

    return 0;
}
