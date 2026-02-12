#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

// Kernel declaration
extern __global__ void Index1D(float* in, float* out, int d);

int main() {
    // Test 1D indexing: each thread copies one element
    const int d = 64;

    // Host arrays
    float h_input[d];
    float h_output[d];
    float expected[d];

    // Initialize: input[i] = i
    for (int i = 0; i < d; i++) {
        h_input[i] = (float)i;
        expected[i] = h_input[i];
    }

    // Device arrays
    float *d_input, *d_output;
    cudaMalloc(&d_input, d * sizeof(float));
    cudaMalloc(&d_output, d * sizeof(float));

    // Copy to device
    cudaMemcpy(d_input, h_input, d * sizeof(float), cudaMemcpyHostToDevice);

    // Launch: d threads, each handles one element
    Index1D<<<1, d>>>(d_input, d_output, d);

    // Copy back
    cudaMemcpy(h_output, d_output, d * sizeof(float), cudaMemcpyDeviceToHost);

    // Validate
    for (int i = 0; i < d; i++) {
        assert(h_output[i] == expected[i]);
    }

    printf("1D Indexing Test Passed!\n");
    printf("Shape: (%d)\n", d);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
