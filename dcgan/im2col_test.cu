// im2col_test.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>

// Define helper macros
#define C10_LAUNCH_BOUNDS_1(x) __launch_bounds__(x)
#define C10_CUDA_KERNEL_LAUNCH_CHECK()                                     \
    {                                                                       \
        cudaError_t err = cudaGetLastError();                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

// Define CUDA_KERNEL_LOOP
#define CUDA_KERNEL_LOOP(i, n)                                           \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);        \
         i += blockDim.x * gridDim.x)

// Define GET_BLOCKS
inline int GET_BLOCKS(int n, int threads_per_block = 1024) {
    return (n + threads_per_block - 1) / threads_per_block;
}

// Optimized im2col kernel (our_im2col)
template <typename dt>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void our_im2col_kernel_blocked(
    const int64_t n, 
    const dt* __restrict__ data_im, 
    const int64_t height,
    const int64_t width,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t dilation_height,
    const int64_t dilation_width,
    const int64_t height_col,
    const int64_t width_col,
    const int64_t block_size, 
    dt* __restrict__ data_col)
{
    // each thread deal with (width_col / block_size) nums of columns 
    const int64_t columns_per_thread = width_col / block_size;

    CUDA_KERNEL_LOOP(index, n) {
        if (index >= n) {
            return;
        }
        const int64_t offset = index % block_size;
        int64_t base = index / block_size;

        const int64_t h_out = base % height_col;
        base /= height_col;

        const int64_t idx_w = base % kernel_width;
        base /= kernel_width;

        const int64_t idx_h = base % kernel_height;
        base /= kernel_height;

        const int64_t channel_in = base;
        const int64_t channel_out = channel_in * kernel_height * kernel_width;

        const int64_t h_in = h_out * stride_height - pad_height;
        const int64_t w_in = -pad_width;

        dt* col_ptr = data_col
            + (height_col * width_col) * channel_out
            + (height_col * width_col) * (idx_h * kernel_width + idx_w)
            + width_col * h_out;

        const dt* im_ptr = data_im + (channel_in * height + h_in) * width + w_in;

        # pragma unroll
        for (int64_t i = 0; i < columns_per_thread; ++i) {
            const int64_t w_out = offset * columns_per_thread + i;

            const int64_t h = h_in + idx_h * dilation_height;
            const int64_t w = w_in + w_out * stride_width + idx_w * dilation_width;

            dt val = static_cast<dt>(0);
            if ((h >= 0) && (w >= 0) && (h < height) && (w < width)) {
                val = im_ptr[idx_h * dilation_height * width
                             + idx_w * dilation_width
                             + w_out * stride_width];
            }
            col_ptr[w_out] = val;
        }
    }
}

// 带 block_size 的 im2col wrapper
template <typename dt>
void our_im2col_blocked(
    cudaStream_t stream,
    const dt* data_im,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t height_col,
    const int64_t width_col,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t dilation_height,
    const int64_t dilation_width,
    const int64_t block_size,  
    dt* data_col)
{
    // num of threads: n = channels * height_col * kernel_height * kernel_width * block_size
    int64_t num_kernels = channels
                        * height_col
                        * kernel_height
                        * kernel_width
                        * block_size;

    int threads = 1024;
    int blocks = GET_BLOCKS(num_kernels, threads);

    our_im2col_kernel_blocked<<<blocks, threads, 0, stream>>>(
        num_kernels,
        data_im,
        height,
        width,
        kernel_height,
        kernel_width,
        pad_height,
        pad_width,
        stride_height,
        stride_width,
        dilation_height,
        dilation_width,
        height_col,
        width_col,
        block_size,
        data_col);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// Original im2col kernel
template <typename dt>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void im2col_kernel(
    const int64_t n,
    const dt* data_im,
    const int64_t height,
    const int64_t width,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t dilation_height,
    const int64_t dilation_width,
    const int64_t height_col,
    const int64_t width_col,
    dt* data_col) 
{
    CUDA_KERNEL_LOOP(index, n) {
        int64_t w_out = index % width_col;
        int64_t idx = index / width_col;
        int64_t h_out = idx % height_col;
        int64_t channel_in = idx / height_col;
        int64_t channel_out = channel_in * kernel_height * kernel_width;
        int64_t h_in = h_out * stride_height - pad_height;
        int64_t w_in = w_out * stride_width - pad_width;

        dt* col = data_col + (channel_out * height_col + h_out) * width_col + w_out;
        const dt* im = data_im + (channel_in * height + h_in) * width + w_in;

        for (int64_t i = 0; i < kernel_height; ++i) {
            for (int64_t j = 0; j < kernel_width; ++j) {
                int64_t h = h_in + i * dilation_height;
                int64_t w = w_in + j * dilation_width;
                *col = (h >= 0 && w >= 0 && h < height && w < width)
                    ? im[i * dilation_height * width + j * dilation_width]
                    : static_cast<dt>(0);
                col += height_col * width_col;
            }
        }
    }
}

// Wrapper for original im2col kernel
template <typename dt>
void im2col(
    cudaStream_t stream,
    const dt* data_im,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t height_col,
    const int64_t width_col,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t dilation_height,
    const int64_t dilation_width,
    dt* data_col) 
{
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int64_t num_kernels = channels * height_col * width_col;
    // printf("num_kernels of origin_im2col: %d\n", num_kernels);
    int threads = 1024;
    int blocks = GET_BLOCKS(num_kernels, threads);
    // Launch CUDA_NUM_THREADS = 1024
    im2col_kernel<<<blocks, threads, 0, stream>>>(
        num_kernels,
        data_im,
        height,
        width,
        kernel_height,
        kernel_width,
        pad_height,
        pad_width,
        stride_height,
        stride_width,
        dilation_height,
        dilation_width,
        height_col,
        width_col,
        data_col);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Utility function to initialize data with random values
void initialize_data(float* data, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Utility function to compare two arrays
bool compare_arrays(const float* a, const float* b, int size, float epsilon=1e-5) {
    for (int i = 0; i < size; ++i) {
        float diff = fabs(a[i] - b[i]);
        if (diff > epsilon) {
            printf("Difference at index %d: a = %f, b = %f, diff = %f\n", i, a[i], b[i], diff);
            return false;
        }
    }
    return true;
}

int main() {
    // Define convolution parameters
    const int64_t channels = 256;
    const int64_t height = 512;
    const int64_t width = 512;

    const int64_t kernel_height = 5;
    const int64_t kernel_width = 5;

    const int64_t pad_height = 1;
    const int64_t pad_width = 2;

    const int64_t stride_height = 2;
    const int64_t stride_width = 2;

    const int64_t dilation_height = 1;
    const int64_t dilation_width = 1;

    // Compute output dimensions
    const int64_t height_col = (height + 2 * pad_height - dilation_height * (kernel_height - 1) - 1) / stride_height + 1;
    const int64_t width_col = (width + 2 * pad_width - dilation_width * (kernel_width - 1) - 1) / stride_width + 1;

    printf("Input dimensions: channels=%ld, height=%ld, width=%ld\n", channels, height, width);
    printf("Kernel dimensions: height=%ld, width=%ld\n", kernel_height, kernel_width);
    printf("Stride: height=%ld, width=%ld\n", stride_height, stride_width);
    printf("Padding: height=%ld, width=%ld\n", pad_height, pad_width);
    printf("Dilation: height=%ld, width=%ld\n", dilation_height, dilation_width);
    printf("Output dimensions: height_col=%ld, width_col=%ld\n", height_col, width_col);

    // Allocate host memory
    size_t input_size = channels * height * width;
    size_t output_size = channels * height_col * width_col * kernel_height * kernel_width;

    float* h_data_im = (float*)malloc(input_size * sizeof(float));
    float* h_data_col_our = (float*)malloc(output_size * sizeof(float));
    float* h_data_col_orig = (float*)malloc(output_size * sizeof(float));

    // Initialize input data
    initialize_data(h_data_im, input_size);

    // Allocate device memory
    float* d_data_im;
    float* d_data_col_our;
    float* d_data_col_orig;

    cudaMalloc((void**)&d_data_im, input_size * sizeof(float));
    cudaMalloc((void**)&d_data_col_our, output_size * sizeof(float));
    cudaMalloc((void**)&d_data_col_orig, output_size * sizeof(float));
    // Copy input data to device
    cudaMemcpy(d_data_im, h_data_im, input_size * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // === Warm-up Phase ===
    // Warm-up iterations to initialize CUDA context and compile kernels
    const int warmup_iterations = 5;
    const int block_size = 100;
    printf("Starting CUDA warm-up (%d iterations)...\n", warmup_iterations);
    for(int i = 0; i < warmup_iterations; ++i){
        // Warm-up our_im2col
        our_im2col_blocked<float>(
            stream,
            d_data_im,
            channels,
            height,
            width,
            height_col,
            width_col,
            kernel_height,
            kernel_width,
            pad_height,
            pad_width,
            stride_height,
            stride_width,
            dilation_height,
            dilation_width,
            block_size,
            d_data_col_our);
    }
    // Wait for all warm-up operations to finish
    cudaStreamSynchronize(stream);
    printf("CUDA warm-up completed.\n");

    // === Timing and Testing Phase ===
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int test_iterations = 500;
    float test_time_our = 0;
    float milliseconds_our = 0;
    float test_time_orig = 0;
    float milliseconds_orig = 0;
    // Launch our_im2col and measure time
    for(int i = 0; i < test_iterations; ++i){
        cudaEventRecord(start, stream);
        cudaEventSynchronize(start);   
        our_im2col_blocked<float>(
            stream,
            d_data_im,
            channels,
            height,
            width,
            height_col,
            width_col,
            kernel_height,
            kernel_width,
            pad_height,
            pad_width,
            stride_height,
            stride_width,
            dilation_height,
            dilation_width,
            block_size,
            d_data_col_our);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds_our, start, stop);
        test_time_our += milliseconds_our;
        milliseconds_our = 0;
    }
    milliseconds_our = test_time_our/test_iterations;
    printf("our_im2col avg execution time: %f ms\n", milliseconds_our);


    printf("Starting CUDA warm-up (%d iterations)...\n", warmup_iterations);
    for(int i = 0; i < warmup_iterations; ++i){
        im2col<float>(
            stream,
            d_data_im,
            channels,
            height,
            width,
            height_col,
            width_col,
            kernel_height,
            kernel_width,
            pad_height,
            pad_width,
            stride_height,
            stride_width,
            dilation_height,
            dilation_width,
            d_data_col_orig);
    }
    // Wait for all warm-up operations to finish
    cudaStreamSynchronize(stream);
    printf("CUDA warm-up completed.\n");

    // Launch original im2col and measure time
    for(int i = 0; i < test_iterations; ++i){
        cudaEventRecord(start, stream);
        cudaEventSynchronize(start);
        im2col<float>(
            stream,
            d_data_im,
            channels,
            height,
            width,
            height_col,
            width_col,
            kernel_height,
            kernel_width,
            pad_height,
            pad_width,
            stride_height,
            stride_width,
            dilation_height,
            dilation_width,
            d_data_col_our);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds_orig, start, stop);
        test_time_orig += milliseconds_orig;
        milliseconds_orig = 0;
    }
    milliseconds_orig = test_time_orig/test_iterations;
    printf("original im2col avg execution time: %f ms\n", milliseconds_orig);

    // Wait for all operations to finish
    cudaStreamSynchronize(stream);

    // Copy results back to host
    cudaMemcpy(h_data_col_our, d_data_col_our, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data_col_orig, d_data_col_orig, output_size * sizeof(float), cudaMemcpyDeviceToHost);


    // Verify that both outputs are the same
    bool our_correct = compare_arrays(h_data_col_our, h_data_col_orig, output_size);
    if (our_correct) {
        printf("SUCCESS: Our kernel produce the same output.\n");
    } else {
        printf("ERROR: Outputs of our kernels do not match.\n");
    }

    // Clean up
    cudaFree(d_data_im);
    cudaFree(d_data_col_our);
    cudaFree(d_data_col_orig);

    free(h_data_im);
    free(h_data_col_our);
    free(h_data_col_orig);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaStreamDestroy(stream);

    return 0;
}