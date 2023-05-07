#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define BMP_HEADER_SIZE 54

__device__ uint8_t secret_key[3] = { 0xAB, 0xCD, 0xEF };

__global__ void encode_image(uint8_t* image, uint32_t width, uint32_t height)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = idx * 3 + idy * width * 3;
    if (idx < width && idy < height) {
        uint8_t r = image[offset];
        uint8_t g = image[offset + 1];
        uint8_t b = image[offset + 2];
        image[offset] = r ^ secret_key[0];
        image[offset + 1] = g ^ secret_key[1];
        image[offset + 2] = b ^ secret_key[2];
    }
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        printf("Usage: %s input_image.bmp\n", argv[0]);
        return 1;
    }

    // Open input image file
    FILE* fp = fopen(argv[1], "rb");
    if (fp == NULL) {
        printf("Failed to open input image file\n");
        return 1;
    }

    // Read BMP header
    uint8_t header[BMP_HEADER_SIZE];
    if (fread(header, 1, BMP_HEADER_SIZE, fp) != BMP_HEADER_SIZE) {
        printf("Failed to read BMP header\n");
        return 1;
    }

    // Read image data
    uint32_t width = *(uint32_t*)(header + 18);
    uint32_t height = *(uint32_t*)(header + 22);
    uint32_t data_size = *(uint32_t*)(header + 34);
    uint8_t* image = (uint8_t*)malloc(data_size);
    if (fread(image, 1, data_size, fp) != data_size) {
        printf("Failed to read image data\n");
        return 1;
    }

    // Close input image file
    fclose(fp);

    // Allocate memory on GPU
    uint8_t* d_image;
    cudaMalloc((void**)&d_image, data_size);

    // Copy image data from CPU to GPU
    cudaMemcpy(d_image, image, data_size, cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 block_size(16, 16);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);

    // Launch kernel to encode image
    encode_image<<<grid_size, block_size>>>(d_image, width, height);

    // Copy encoded image data from GPU to CPU
    cudaMemcpy(image, d_image, data_size, cudaMemcpyDeviceToHost);

    // Open output image file
    fp = fopen("encoded_image.bmp", "wb");
    if (fp == NULL) {
        printf("Failed to open output image file\n");
        return 1;
    }

    // Write BMP header
    if (fwrite(header, 1, BMP_HEADER_SIZE, fp) != BMP_HEADER_SIZE) {
        printf("Failed to write BMP header\n");
        return 1;
    }

    // Write encoded image data
    if (fwrite(image, 1, data_size, fp) != data_size) {
        printf("Failed to write encoded image data\n");
        return 1;
    }

    // Close output image file
    fclose(fp);

    // Free memory on GPU and CPU
    cudaFree(d_image);
    free(image);

    return 0;
}
