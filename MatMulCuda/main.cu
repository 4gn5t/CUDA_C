#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <string.h>
#include <math.h>
#include <time.h>

inline void checkCudaErrors(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd = aBegin + wA - 1;
    int aStep = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;
    int bStep = BLOCK_SIZE * wB;

    float Csub = 0;

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
    {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

// Generate matrices
void generateMatrix(float *A, float *B, int m, int k, int n) {
    srand(time(NULL));
    for (int i = 0; i < m * k; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < k * n; ++i) {
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Sequential 
void sequentialMatrixMultiplication(float *C, float *A, float *B, int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0;
            for (int p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void parallelMatrixMultiplicationCUDA(float *C, float *A, float *B, int m, int k, int n) {
    float *d_A, *d_B, *d_C;

    checkCudaErrors(cudaMalloc((void **)&d_A, m * k * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_B, k * n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_C, m * n * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice));

    int block_size = 32;
    dim3 threads(block_size, block_size);
    dim3 grid((n + block_size - 1) / block_size, (m + block_size - 1) / block_size);

    MatrixMulCUDA<32><<<grid, threads>>>(d_C, d_A, d_B, k, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void parallelMatrixMultiplicationCuBLAS(float *C, float *A, float *B, int m, int k, int n) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void **)&d_A, m * k * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_B, k * n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_C, m * n * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_A, A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, B, k * n * sizeof(float), cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);

    checkCudaErrors(cudaMemcpy(C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}

int main(int argc, char **argv) {
    int m = 1000, k = 1000, n = 1000; // Matrix dimensions
    printf("Matrix dimensions: %d x %d x %d\n", m, k, n);

    float *h_A = (float *)malloc(m * k * sizeof(float));
    float *h_B = (float *)malloc(k * n * sizeof(float));
    float *h_C_seq = (float *)malloc(m * n * sizeof(float));
    float *h_C_cuda = (float *)malloc(m * n * sizeof(float));
    float *h_C_cublas = (float *)malloc(m * n * sizeof(float));

    generateMatrix(h_A, h_B, m, k, n);

    clock_t start = clock();
    sequentialMatrixMultiplication(h_C_seq, h_A, h_B, m, k, n);
    clock_t end = clock();
    printf("Sequential multiplication time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    start = clock();
    parallelMatrixMultiplicationCUDA(h_C_cuda, h_A, h_B, m, k, n); // Implemented earlier
    end = clock();
    printf("CUDA multiplication time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    start = clock();
    parallelMatrixMultiplicationCuBLAS(h_C_cublas, h_A, h_B, m, k, n);
    end = clock();
    printf("cuBLAS multiplication time: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    free(h_A);
    free(h_B);
    free(h_C_seq);
    free(h_C_cuda);
    free(h_C_cublas);

    return 0;
}