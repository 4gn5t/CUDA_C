#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <string.h>
#include <math.h>
#include <time.h>

void saxpy(int n, float *x, float *y, float a);

void saxpy(int n, float *x, float *y, float a)
{
    int i;
    for (i = 0; i < n; i++)
    {
        y[i] = a * x[i] + y[i];
    }
}

// CUDA kernel for SAXPY
#ifdef __CUDACC__
__global__ void saxpy_kernel(int n, float a, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
#endif

// Host wrapper for CUDA SAXPY
void saxpy_cuda(int n, float a, float *x, float *y) {
    float *d_x = NULL, *d_y = NULL;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

#ifdef __CUDACC__
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    saxpy_kernel<<<numBlocks, blockSize>>>(n, a, d_x, d_y);
    cudaDeviceSynchronize();
#endif
    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
}

// cuBLAS SAXPY
void saxpy_cublas(int n, float a, float *x, float *y) {
    float *d_x = NULL, *d_y = NULL;
    cublasHandle_t handle;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

    cublasCreate(&handle);
    // y = a*x + y
    cublasSaxpy(handle, n, &a, d_x, 1, d_y, 1);

    cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    cublasDestroy(handle);
    cudaFree(d_x);
    cudaFree(d_y);
}

void fill_arrays(float *x, float *y, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = (float)i;
        y[i] = (float)i;
    }
}

void print_head(float *x, float *y, int n) {
    for (int i = 0; i < (n < 10 ? n : 10); i++) {
        printf("x[%d] = %f, y[%d] = %f\n", i, x[i], i, y[i]);
    }
}

int main() {
    int sizes[] = {10000, 100000, 1000000, 10000000, 100000000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    float a = 2.0f;

    printf("|   N   |  CPU SAXPY (s) | CUDA SAXPY (s) | cuBLAS SAXPY (s) |\n");
    printf("|-------|----------------|----------------|------------------|\n");

    for (int idx = 0; idx < num_sizes; ++idx) {
        int n = sizes[idx];
        size_t size = n * sizeof(float);
        float *x = (float*)malloc(size);
        float *y = (float*)malloc(size);
        float *y_cuda = (float*)malloc(size);
        float *y_cublas = (float*)malloc(size);

        if (!x || !y || !y_cuda || !y_cublas) {
            printf("Memory allocation failed");
            if (x) free(x);
            if (y) free(y);
            if (y_cuda) free(y_cuda);
            if (y_cublas) free(y_cublas);
            continue;
        }

        fill_arrays(x, y, n);
        memcpy(y_cuda, y, size);
        memcpy(y_cublas, y, size);

        // CPU SAXPY
        clock_t start = clock();
        saxpy(n, x, y, a);
        clock_t end = clock();
        double cpu_time = (double)(end - start) / CLOCKS_PER_SEC;

        // CUDA SAXPY
        start = clock();
        saxpy_cuda(n, a, x, y_cuda);
        end = clock();
        double cuda_time = (double)(end - start) / CLOCKS_PER_SEC;

        // cuBLAS SAXPY
        start = clock();
        saxpy_cublas(n, a, x, y_cublas);
        end = clock();
        double cublas_time = (double)(end - start) / CLOCKS_PER_SEC;

        printf("|%7d|%16.6f|%16.6f|%18.6f|\n", n, cpu_time, cuda_time, cublas_time);

        if (n == sizes[0]) {
            // printf("Sample output for n=%d:\n", n);
            // print_head(x, y, n);
        }

        free(x);
        free(y);
        free(y_cuda);
        free(y_cublas);
    }
    return 0;
}