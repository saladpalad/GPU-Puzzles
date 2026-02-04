#include <stdio.h>

extern __global__ void matmul_guards(float *A, float *B, float *C, int M, int N, int K);

void run_kernel(){
  int M = 1025;
  int N = 2048;
  int K = 2048;
  int BLOCK_M = 16; // num_threads along M
  int BLOCK_N = 16; // num_threads along N
  cudaEvent_t start, end;
  float ms;
  float *A, *B, *C;
  cudaMallocManaged(&A, M*K*sizeof(float));
  cudaMallocManaged(&B, K*N*sizeof(float));
  cudaMallocManaged(&C, M*N*sizeof(float));
  cudaMemset(C, 0, M*N*sizeof(float));
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  for (int i = 0; i < M*N; i++){
    A[i] = static_cast<float>(i);
  }

  for (int i = 0; i < K; i++){
    for (int j = 0; j < N; j++){
      B[i*N+j] = (i == j) ? 1.0f : 0.0f;
    }
  }

  int m_blocks = (M + (BLOCK_M-1)) / BLOCK_M;
  int n_blocks = (N + (BLOCK_N-1)) / BLOCK_N;
  dim3 grid_dim(m_blocks, n_blocks, 1);
  dim3 block_dim(BLOCK_M, BLOCK_N, 1);
  cudaEventRecord(start);
  matmul_guards<<<grid_dim, block_dim>>>(A, B, C, M, N, K);
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&ms, start, end);
  //cudaDeviceSynchronize();
  bool fail = false;
  for (int i = 0; i < M*N; i++){
    if (C[i] != A[i]){
      fail = true;
      printf("Error at index %i\n", i);
      break;
    }
  }
  fail ? printf("test failed\n") : printf("test passed\n");
  if (!fail) printf("matmul kernel took %f ms\n", ms);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}

int main(){
  run_kernel();
  return 0;
}