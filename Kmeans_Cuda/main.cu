#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

/*
    * KMeans clustering algorithm 
    * Randomly generate clusters 
    * Calculate Euclidean distance for each point to each centroid -> sqrt(sum(a_i - b_i)^2)
    * Assign each point to the nearest centroid -> labels[i] = best
    * Update centroids by averaging the points assigned to each centroid -> centroids[c * dim + d] = new_centroids[c * dim + d] / counts[c]
    * Repeat until convergence or max iterations -> max_iters
*/

// Randomly generate clusters
void generate_clusters(float *data, int num_points, int num_clusters, int dim, float spread) {
    srand(time(NULL));
    for (int i = 0; i < num_points; i++) {
        int cluster = rand() % num_clusters;
        for (int j = 0; j < dim; j++) {
            data[i * dim + j] = (rand() % 100) / 100.0f + cluster * spread;
        }
    }
}

void write_data_to_file(const char *filename, float *data, int num_points, int dim) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file for writing\n");
        return;
    }
    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < dim; j++) {
            fprintf(file, "%f ", data[i * dim + j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

// Euclidean distance sqrt(a^2 + b^2) = sqrt(sum(a_i - b_i)^2)
float euclidean_distance_CPU(const float *a, const float *b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = a[i] - b[i]; // a[i] - b[i]
        sum += diff * diff; // (a[i] - b[i])^2
    }
    return sqrtf(sum); // sqrt(sum(a_i - b_i)^2)
}

// CPU KMeans 
void kmeans_cpu(const float *data, int *labels, float *centroids, int num_points, int num_clusters, int dim, int max_iters) {
    float *new_centroids = (float *)calloc(num_clusters * dim, sizeof(float));
    int *counts = (int *)calloc(num_clusters, sizeof(int));

    for (int iter = 0; iter < max_iters; iter++) { // iter until convergence or max iterations
        for (int i = 0; i < num_points; i++) { // for each point
            float min_dist =FLT_MAX; // min distance (FLT_MAX -> float max value)
            int best= -1;
            for (int c =0; c < num_clusters; c++) {
                float dist = euclidean_distance_CPU(&data[i * dim], &centroids[c * dim], dim); // Euclidean distance for each point to each centroid
                if (dist < min_dist) { // if the distance is less than the minimum distance
                    min_dist= dist; // Update the minimum distance
                    best=c; // Update the best cluster
                }
            }
            labels[i] = best; // Assign each point to the nearest centroid
        }

        // Update step
        memset(new_centroids, 0, num_clusters * dim * sizeof(float)); // reset new centroids
        memset(counts, 0, num_clusters * sizeof(int)); // reset counts

        // sum the points assigned to each centroid
        for (int i = 0; i < num_points; i++) { // for each point
            int c = labels[i];
            counts[c]++;
            for (int d = 0; d < dim; d++) { // for each dimension
                new_centroids[c * dim + d] += data[i * dim + d]; // sum the points assigned to each centroid
            }
        }

        // average the points assigned to each centroid
        for (int c = 0; c < num_clusters; c++) { // for each cluster
            if (counts[c] > 0) { // if the count is highier than 0
                for (int d = 0; d < dim; d++) { // for each dimension
                    centroids[c * dim+d] = new_centroids[c*dim + d] / counts[c]; // average the points assigned to each centroid
                }
            }
        }
    }
    free(new_centroids);
    free(counts);
}

// CUDA Euclidean distance
__device__ float euclidean_distance_CUDA(const float *a, const float *b, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

// CUDA assign labels
__global__ void assign_labels_cuda(const float *data, const float *centroids, int *labels, int num_points, int num_clusters, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_points) {
        float min_dist = FLT_MAX;
        int best = -1;
        for (int c = 0; c < num_clusters; c++) {
            float dist = euclidean_distance_CUDA(&data[idx * dim], &centroids[c * dim], dim);
            if (dist < min_dist) {
                min_dist = dist;
                best = c;
            }
        }
        labels[idx] = best;
    }
}

// CUDA update centroids
__global__ void update_centroids_cuda(const float *data, float *centroids, const int *labels, int *counts, int num_points, int num_clusters, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        int c = labels[idx];
        atomicAdd(&counts[c], 1);
        for (int d = 0; d < dim; d++) {
            atomicAdd(&centroids[c * dim + d], data[idx * dim + d]);
        }
    }
}

// CUDA KMeans wrapper
void kmeans_cuda(const float *h_data, int *h_labels, float *h_centroids, int num_points, int num_clusters, int dim, int max_iters) {
    // Device pointers for data, centroids, labels, and counts
    float *d_data, *d_centroids;
    int *d_labels, *d_counts;

    // Calculate memory sizes for arrays
    size_t data_size = num_points * dim * sizeof(float);
    size_t centroids_size = num_clusters * dim * sizeof(float);
    size_t labels_size = num_points * sizeof(int);
    size_t counts_size = num_clusters * sizeof(int);

    // Allocate memory on the GPU for all arrays
    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_centroids, centroids_size);
    cudaMalloc(&d_labels, labels_size);
    cudaMalloc(&d_counts, counts_size);

    // Copy input data and initial centroids from host (CPU) to device (GPU)
    cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, centroids_size, cudaMemcpyHostToDevice);

    // Set up CUDA kernel launch configuration
    int threads = 256;
    int blocks_points = (num_points + threads - 1) / threads;

    // Allocate temporary arrays on the host for centroid sums and counts
    float *tmp_centroids = (float *)malloc(centroids_size);
    int *tmp_counts = (int *)malloc(counts_size);

    // Main KMeans iteration loop
    for (int iter = 0; iter < max_iters; iter++) {
        // Assign each point to the nearest centroid (on GPU)
        assign_labels_cuda<<<blocks_points, threads>>>(d_data, d_centroids, d_labels, num_points, num_clusters, dim);
        cudaDeviceSynchronize(); // Wait for kernel to finish
        // Clear centroids and counts on device before accumulation
        cudaMemset(d_centroids, 0, centroids_size);
        cudaMemset(d_counts, 0, counts_size);
        // Accumulate sums for new centroids and count points per cluster (on GPU)
        update_centroids_cuda<<<blocks_points, threads>>>(d_data, d_centroids, d_labels, d_counts, num_points, num_clusters, dim);
        cudaDeviceSynchronize(); // Wait for kernel to finish
        // Copy centroid sums and counts from device to host
        cudaMemcpy(tmp_centroids, d_centroids, centroids_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(tmp_counts, d_counts, counts_size, cudaMemcpyDeviceToHost);

        // Calculate the mean for each centroid on the host
        for (int c = 0; c < num_clusters; c++) {
            if (tmp_counts[c] > 0) {
                for (int d = 0; d < dim; d++) {
                    tmp_centroids[c * dim + d] /= tmp_counts[c];
                }
            }
        }

        // Copy updated centroids back to device for the next iteration
        cudaMemcpy(d_centroids, tmp_centroids, centroids_size, cudaMemcpyHostToDevice);
    }

    // Copy final labels and centroids from device to host
    cudaMemcpy(h_labels, d_labels, labels_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centroids, d_centroids, centroids_size, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_counts);
    free(tmp_centroids);
    free(tmp_counts);
}


// write cluster assign to file
void write_labels_to_file(const char *filename, const int *labels, int num_points) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file for writing labels\n");
        return;
    }
    for (int i = 0; i < num_points; i++) {
        fprintf(file, "%d\n", labels[i]);
    }
    fclose(file);
}

// write centroids to file
void write_centroids_to_file(const char *filename, const float *centroids, int num_clusters, int dim) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file for writing centroids\n");
        return;
    }
    for (int c = 0; c < num_clusters; c++) {
        for (int d = 0; d < dim; d++) {
            fprintf(file, "%f ", centroids[c * dim + d]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

int main(int argc, char **argv) {
    int num_points = 300 ;
    int num_clusters = 3;
    int dim = 2;
    float spread = 1.0;
    int max_iters = 50;


    float *data = (float *)malloc(num_points * dim * sizeof(float));
    generate_clusters(data, num_points, num_clusters, dim, spread);
    write_data_to_file("clusters.txt", data, num_points, dim);

    // CPU KMeans 
    float *centroids_cpu = (float *)malloc(num_clusters * dim * sizeof(float));
    int *labels_cpu = (int *)malloc(num_points * sizeof(int));
    
    // Init centroids randomly from data
    for (int c = 0; c < num_clusters; c++) {
        memcpy(&centroids_cpu[c * dim], &data[(c * num_points / num_clusters) * dim], dim * sizeof(float));
    }
    
    // Time CPU KMeans
    clock_t start_cpu = clock();
    kmeans_cpu(data, labels_cpu, centroids_cpu, num_points, num_clusters, dim, max_iters);
    clock_t end_cpu = clock();
    double cpu_time = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    printf("CPU KMeans time: %.4f seconds\n", cpu_time);

    // Write results to files
    write_labels_to_file("labels_cpu.txt", labels_cpu, num_points);
    write_centroids_to_file("centroids_cpu.txt", centroids_cpu, num_clusters, dim);

    // CUDA KMeans
    float *centroids_cuda =(float *)malloc(num_clusters * dim * sizeof(float));
    int *labels_cuda =(int *)malloc(num_points * sizeof(int));
    memcpy(centroids_cuda,centroids_cpu, num_clusters * dim * sizeof(float));

    // Time CUDA KMeans
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kmeans_cuda(data, labels_cuda, centroids_cuda, num_points, num_clusters, dim, max_iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cuda_time = 0;
    cudaEventElapsedTime(&cuda_time, start, stop);
    printf("CUDA KMeans time: %.4f seconds\n", cuda_time / 1000.0f);

    // Write results to files
    write_labels_to_file("labels_cuda.txt", labels_cuda, num_points);
    write_centroids_to_file("centroids_cuda.txt", centroids_cuda, num_clusters, dim);

    // Cleanup
    free(data);
    free(centroids_cpu);
    free(labels_cpu);
    free(centroids_cuda);
    free(labels_cuda);

    return 0;
}