#include <torch/torch.h>
#include <iostream>
#include <chrono>

#define BLOCKSIZE 16

// Custom GEMM kernel
template <typename scalar_t>
__global__ void gemm_kernel(
    const scalar_t* A, const scalar_t* B, scalar_t* C,
    int64_t M, int64_t N, int64_t K) {
  __shared__ scalar_t tile_A[BLOCKSIZE][BLOCKSIZE];
  __shared__ scalar_t tile_B[BLOCKSIZE][BLOCKSIZE];

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  scalar_t sum = 0;

  for (int t = 0; t < (K + BLOCKSIZE - 1) / BLOCKSIZE; ++t) {
    if (row < M && t * BLOCKSIZE + threadIdx.x < K)
      tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * BLOCKSIZE + threadIdx.x];
    else
      tile_A[threadIdx.y][threadIdx.x] = 0;

    if (col < N && t * BLOCKSIZE + threadIdx.y < K)
      tile_B[threadIdx.y][threadIdx.x] = B[(t * BLOCKSIZE + threadIdx.y) * N + col];
    else
      tile_B[threadIdx.y][threadIdx.x] = 0;

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < BLOCKSIZE; ++i) {
      sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] += sum;
  }
}

// Custom GEMM wrapper
template <typename scalar_t>
void custom_gemm(
    int64_t M, int64_t N, int64_t K,
    scalar_t alpha, const scalar_t* A, int64_t lda,
    const scalar_t* B, int64_t ldb,
    scalar_t beta, scalar_t* C, int64_t ldc) {
  dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
  dim3 gridDim((N + BLOCKSIZE - 1) / BLOCKSIZE, (M + BLOCKSIZE - 1) / BLOCKSIZE);

  gemm_kernel<<<gridDim, blockDim, 0>>>(
      A, B, C, M, N, K);
  // C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// CPU GEMM implementation
template <typename scalar_t>
void cpu_gemm(
    int64_t M, int64_t N, int64_t K,
    scalar_t alpha, const scalar_t* A, int64_t lda,
    const scalar_t* B, int64_t ldb,
    scalar_t beta, scalar_t* C, int64_t ldc) {
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      scalar_t sum = 0;
      for (int64_t k = 0; k < K; ++k) {
        sum += A[i * lda + k] * B[k * ldb + j];
      }
      C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
    }
  }
}

// Test function
void test_gemm(int64_t M, int64_t N, int64_t K) {
  // Create random matrices on CPU
  auto A_cpu = torch::rand({M, K}, torch::kFloat32);
  auto B_cpu = torch::rand({K, N}, torch::kFloat32);
  auto C_cpu = torch::zeros({M, N}, torch::kFloat32);

  // Copy matrices to GPU
  auto A_gpu = A_cpu.to(torch::kCUDA);
  auto B_gpu = B_cpu.to(torch::kCUDA);
  auto C_gpu = C_cpu.to(torch::kCUDA);

  // Perform CPU GEMM
  auto start_cpu = std::chrono::high_resolution_clock::now();
  cpu_gemm<float>(
      M, N, K,
      1.0f, A_cpu.data_ptr<float>(), K,
      B_cpu.data_ptr<float>(), N,
      0.0f, C_cpu.data_ptr<float>(), N);
  auto end_cpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
  std::cout << "CPU GEMM time: " << cpu_duration.count() << " seconds\n";

  // Perform GPU GEMM
  auto start_gpu = std::chrono::high_resolution_clock::now();
  custom_gemm<float>(
      M, N, K,
      1.0f, A_gpu.data_ptr<float>(), K,
      B_gpu.data_ptr<float>(), N,
      0.0f, C_gpu.data_ptr<float>(), N);
  cudaDeviceSynchronize(); // Ensure GPU computation is complete
  auto end_gpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;
  std::cout << "GPU GEMM time: " << gpu_duration.count() << " seconds\n";

  // Copy GPU result back to CPU
  auto C_gpu_cpu = C_gpu.to(torch::kCPU);

  // Compare CPU and GPU results
  bool equal = torch::allclose(C_cpu, C_gpu_cpu, 1e-4, 1e-4);
  if (equal) {
    std::cout << "[[[   Results are equal.   ]]]\n";
  } else {
    std::cerr << "[[[   Results are NOT equal.   ]]]\n";
  }
}

int main() {
  // Test with small matrices
  std::cout << "Testing with small matrices (M=16, N=16, K=16):\n";
  test_gemm(16, 16, 16);

  // Test with larger matrices
  std::cout << "\nTesting with larger matrices (M=1024, N=1024, K=1024):\n";
  test_gemm(1024, 1024, 1024);

  return 0;
}