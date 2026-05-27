#include <cuda_runtime.h>

__global__ void upload_rgb_to_rgba_kernel(cudaSurfaceObject_t surface, const float *rgb, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float *src = rgb + (static_cast<size_t>(y) * width + x) * 3;
    surf2Dwrite(make_float4(src[0], src[1], src[2], 1.0f), surface, x * static_cast<int>(sizeof(float4)), y);
}

cudaError_t upload_rgb_to_rgba_array(cudaArray_t dst, const float *rgb, int width, int height, cudaStream_t stream) {
    cudaResourceDesc desc = {};
    desc.resType = cudaResourceTypeArray;
    desc.res.array.array = dst;

    cudaSurfaceObject_t surface = 0;
    cudaError_t err = cudaCreateSurfaceObject(&surface, &desc);
    if (err != cudaSuccess) {
        return err;
    }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    upload_rgb_to_rgba_kernel<<<grid, block, 0, stream>>>(surface, rgb, width, height);
    err = cudaGetLastError();

    cudaError_t destroy_err = cudaDestroySurfaceObject(surface);
    return err != cudaSuccess ? err : destroy_err;
}
