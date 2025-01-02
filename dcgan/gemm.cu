#include <torch/torch.h>
#include <iostream>
#include <chrono>

#define BLOCKSIZE 8
#define VECTOR_SIZE 4 // Use float4 for vectorized access (4 elements per transaction)

template <typename scalar_t>
__global__ void gemm_kernel(
    const scalar_t* A, const scalar_t* B, scalar_t* C,
    int64_t M, int64_t N, int64_t K) {
  // Shared memory for tiles of A and B
  __shared__ scalar_t tile_A[BLOCKSIZE][BLOCKSIZE * VECTOR_SIZE];
  __shared__ scalar_t tile_B[BLOCKSIZE][BLOCKSIZE * VECTOR_SIZE];

  // Thread indices
  int row = blockIdx.y * BLOCKSIZE + threadIdx.y;
  int col = blockIdx.x * BLOCKSIZE + threadIdx.x;

  // Accumulator for the result
  scalar_t sum[VECTOR_SIZE] = {0};

  // Loop over tiles of A and B
  #pragma unroll
  for (int t = 0; t < (K + BLOCKSIZE * VECTOR_SIZE - 1) / (BLOCKSIZE * VECTOR_SIZE); ++t) {
    // Load a tile of A into shared memory
    int a_col = t * BLOCKSIZE * VECTOR_SIZE + threadIdx.x * VECTOR_SIZE;
    if (row < M) {
      #pragma unroll
      for (int v = 0; v < VECTOR_SIZE; ++v) {
        if (a_col + v < K) {
          tile_A[threadIdx.y][threadIdx.x * VECTOR_SIZE + v] = A[row * K + a_col + v];
        } else {
          tile_A[threadIdx.y][threadIdx.x * VECTOR_SIZE + v] = 0;
        }
      }
    } else {
      #pragma unroll
      for (int v = 0; v < VECTOR_SIZE; ++v) {
        tile_A[threadIdx.y][threadIdx.x * VECTOR_SIZE + v] = 0;
      }
    }

    // Load a tile of B into shared memory
    int b_row = t * BLOCKSIZE * VECTOR_SIZE + threadIdx.y * VECTOR_SIZE;
    if (col < N) {
      #pragma unroll
      for (int v = 0; v < VECTOR_SIZE; ++v) {
        if (b_row + v < K) {
          tile_B[threadIdx.y][threadIdx.x * VECTOR_SIZE + v] = B[(b_row + v) * N + col];
        } else {
          tile_B[threadIdx.y][threadIdx.x * VECTOR_SIZE + v] = 0;
        }
      }
    } else {
      #pragma unroll
      for (int v = 0; v < VECTOR_SIZE; ++v) {
        tile_B[threadIdx.y][threadIdx.x * VECTOR_SIZE + v] = 0;
      }
    }

    __syncthreads();

    // Compute the partial sum for this tile
    for (int i = 0; i < BLOCKSIZE; ++i) {
      #pragma unroll
      for (int v = 0; v < VECTOR_SIZE; ++v) {
        sum[v] += tile_A[threadIdx.y][i * VECTOR_SIZE + v] * tile_B[i][threadIdx.x * VECTOR_SIZE + v];
      }
    }

    __syncthreads();
  }

  // Write the result to C
  if (row < M && col < N) {
    for (int v = 0; v < VECTOR_SIZE; ++v) {
      C[row * N + col] += sum[v];
    }
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
      C[i * ldc + j] += alpha * sum + beta * C[i * ldc + j];
    }
  }
}

// Function to measure average execution time in nanoseconds
// template <typename Func>
// double measure_time_ns(Func func, int repeats = 10) {
//     double total_duration = 0;
//     for (int i = 0; i < repeats; ++i) {
//         auto start = std::chrono::high_resolution_clock::now();
//         func();
//         auto end = std::chrono::high_resolution_clock::now();
//         total_duration += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
//     }
//     return total_duration / repeats; // Return average time in nanoseconds
// }

double measure_time_ns(const std::function<void()>& func, int repeat=10) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i =0;i<repeat;i++){
      func();
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double nanoseconds = milliseconds * 1e6;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return nanoseconds;
}

void test_gemm(int64_t M, int64_t N, int64_t K) {
    // Create random matrices on CPU
    auto A_cpu = torch::rand({M, K}, torch::kFloat32);
    auto B_cpu = torch::rand({K, N}, torch::kFloat32);
    auto C_cpu = torch::zeros({M, N}, torch::kFloat32);

    // Copy matrices to GPU
    cudaSetDevice(1); // Use cuda:1
    auto A_gpu = A_cpu.to(torch::kCUDA);
    auto B_gpu = B_cpu.to(torch::kCUDA);
    auto C_gpu = C_cpu.to(torch::kCUDA);

    // Perform CPU GEMM
    double cpu_time_ns = measure_time_ns([&]() {
        cpu_gemm<float>(
            M, N, K,
            1.0f, A_cpu.data_ptr<float>(), K,
            B_cpu.data_ptr<float>(), N,
            0.0f, C_cpu.data_ptr<float>(), N);
    });
    std::cout << "CPU GEMM average time: " << cpu_time_ns << " ns\n";

    // Perform GPU GEMM
    double gpu_time_ns = measure_time_ns([&]() {
        custom_gemm<float>(
            M, N, K,
            1.0f, A_gpu.data_ptr<float>(), K,
            B_gpu.data_ptr<float>(), N,
            0.0f, C_gpu.data_ptr<float>(), N);
        cudaDeviceSynchronize(); // Ensure GPU computation is complete
    });
    std::cout << "GPU GEMM average time: " << gpu_time_ns << " ns\n";

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
  std::cout << "Testing with small matrices (M=19, N=37, K=71):\n";
  test_gemm(19, 37, 71);

  // Test with larger matrices
  std::cout << "\nTesting with larger matrices (M=1024, N=512, K=512):\n";
  test_gemm(1024, 512, 512);

  return 0;
}