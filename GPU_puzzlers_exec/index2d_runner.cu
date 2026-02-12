#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

// Kernel declaration
extern __global__ void Index2D(float* in, float* out, int n, int d);

int main() {
    // Test 2D indexing: threadIdx.y, threadIdx.x map to (row, col)
    const int n = 16;  // rows (threadIdx.y)
    const int d = 64;  // cols (threadIdx.x) // total: 16*64=1024 threads

    // Host arrays
    float h_input[n * d];
    float h_output[n * d];
    float expected[n * d];

    // Initialize: input[i][j] = i*d + j
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            h_input[i * d + j] = (float)(i * d + j);
            expected[i * d + j] = h_input[i * d + j];
        }
    }

    // Device arrays
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * d * sizeof(float));
    cudaMalloc(&d_output, n * d * sizeof(float));

    // Copy to device
    cudaMemcpy(d_input, h_input, n * d * sizeof(float), cudaMemcpyHostToDevice);

    // Launch: 2D block of threads (n, d)
    dim3 threads(d, n);  // x=d, y=n
    Index2D<<<1, threads>>>(d_input, d_output, n, d);

    // Copy back
    cudaMemcpy(h_output, d_output, n * d * sizeof(float), cudaMemcpyDeviceToHost);

    // Validate
    for (int i = 0; i < n * d; i++) {
        assert(h_output[i] == expected[i]);
    }

    printf("2D Indexing Test Passed!\n");
    printf("Shape: (%d, %d)\n", n, d);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
