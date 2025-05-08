#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

int cuGPUScan()
{
    struct cudaDeviceProp prop;
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0)
    {
        printf("CUDA ERROR::No CUDA Device\n");
        return -1;
    }
    else
    {
        for (int i = 0; i < count; ++i)
        {
            if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) {
                printf("CUDA ERROR::Failed to get properties for device %d\n", i);
                continue;
            }
            printf("---- General Information for device ID:%d ----\n", i);
            printf("Name: %s\n", prop.name);
            printf("Compute capability: %d.%d\n", prop.major, prop.minor);
            printf("Clock rate: %d\n", prop.clockRate);
            printf("Device copy overlap: ");
            if (prop.deviceOverlap)
            {
                printf("Enabled\n");
            }
            else
            {
                printf("Disabled\n");
            }
            printf("Kernel execution timeout: ");
            if (prop.kernelExecTimeoutEnabled)
            {
                printf("Enabled\n");
            }
            else
            {
                printf("Disabled\n");
            }
            printf("---- Memory Information for device ID:%d ----\n", i);
            printf("Total global mem: %zu\n", prop.totalGlobalMem);
            printf("Total constant Mem: %zu\n", prop.totalConstMem);
            printf("Max mem pitch: %zu\n", prop.memPitch);
            printf("Texture Alignment: %zu\n", prop.textureAlignment);
            printf("---- MP Information for device ID:%d ----\n", i);
            printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
            printf("Shared mem per mp: %zu\n", prop.sharedMemPerBlock);
            printf("Registers per mp: %d\n", prop.regsPerBlock);
            printf("Threads in warp: %d\n", prop.warpSize);
            printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
            printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0],
                   prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
            printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0],
                   prop.maxGridSize[1], prop.maxGridSize[2]);
        }
        printf("\n");
    }
    return count;
}

int main()
{
    int count = cuGPUScan();
    if (count < 0)
    {
        printf("CUDA ERROR::No CUDA Device\n");
        return -1;
    }
    else
    {
        printf("CUDA INFO::%d CUDA Device(s) found\n", count);
    }
    printf("CUDA INFO::GPU scan completed\n");
    return 0;
}