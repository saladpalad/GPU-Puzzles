#include <stdio.h>

extern __global__ void Transpose(float *in, float *out, int M, int N);

void runKernel(){
  float *in, *out;
  int N = 256;
  int M = 512;
  int BLOCK_M = 32;
  int BLOCK_N = 32;

  cudaMallocManaged(&in, M*N*sizeof(float));
  cudaMallocManaged(&out, M*N*sizeof(float));

  for(int i = 0; i < M*N; i++){
    in[i] = i;
  }

  cudaMemset(out, 0, M*N*sizeof(float));

  int num_blocks_m = (M + BLOCK_M-1) / BLOCK_M;
  int num_blocks_n = (N + BLOCK_N-1) / BLOCK_N;
  dim3 grid_dim(num_blocks_n, num_blocks_m, 1);
  dim3 block_dim(BLOCK_M, BLOCK_N, 1);
  Transpose<<<grid_dim, block_dim>>>(in, out, M, N);
  cudaDeviceSynchronize();
  bool fail = false;
  for (int i = 0; i < N; i++){
    for (int j = 0; j < M; j++){
      if (out[i*M+j] != in[j*N+i]){
        fail = true;
        printf("error at idx %i\n", i);
        break;
      }
    }
  }
  fail ? printf("test failed\n") : printf("test passed\n");
}

int main(){
  runKernel();
  return 0;
}