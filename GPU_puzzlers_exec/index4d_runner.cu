#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

// Kernel declaration
extern __global__ void Index4D(float* in, float* out, int b, int h, int n, int d);

int main() {
    // Test 4D indexing: blockIdx.y for b, blockIdx.x for h, threadIdx.y for n, threadIdx.x for d
    const int b = 4;   // batch (blockIdx.y)
    const int h = 20;  // heads (blockIdx.x)
    const int n = 16;  // seq_len (threadIdx.y)
    const int d = 64;  // dim (threadIdx.x) // total: 16*64=1024 threads/block

    // Host arrays
    float h_input[b * h * n * d];
    float h_output[b * h * n * d];
    float expected[b * h * n * d];

    // Initialize: input[i][j][k][l] = i*h*n*d + j*n*d + k*d + l
    for (int i = 0; i < b; i++) {
        for (int j = 0; j < h; j++) {
            for (int k = 0; k < n; k++) {
                for (int l = 0; l < d; l++) {
                    int idx = i * h * n * d + j * n * d + k * d + l;
                    h_input[idx] = (float)idx;
                    expected[idx] = h_input[idx];
                }
            }
        }
    }

    // Device arrays
    float *d_input, *d_output;
    cudaMalloc(&d_input, b * h * n * d * sizeof(float));
    cudaMalloc(&d_output, b * h * n * d * sizeof(float));

    // Copy to device
    cudaMemcpy(d_input, h_input, b * h * n * d * sizeof(float), cudaMemcpyHostToDevice);

    // Launch: (h, b) blocks, each block has (d, n) threads
    dim3 blocks(h, b);    // x=h, y=b
    dim3 threads(d, n);   // x=d, y=n
    Index4D<<<blocks, threads>>>(d_input, d_output, b, h, n, d);

    // Copy back
    cudaMemcpy(h_output, d_output, b * h * n * d * sizeof(float), cudaMemcpyDeviceToHost);

    // Validate
    for (int i = 0; i < b * h * n * d; i++) {
        assert(h_output[i] == expected[i]);
    }

    printf("4D Indexing Test Passed!\n");
    printf("Shape: (%d, %d, %d, %d)\n", b, h, n, d);
    printf("Grid: (%d, %d), Block: (%d, %d)\n", h, b, d, n);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
