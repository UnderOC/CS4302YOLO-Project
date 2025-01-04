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
__global__ void our_im2col_kernel(
    const int64_t n,
    const dt* data_im, // shape = [channels, height, width]
    const int64_t height, // input image height
    const int64_t width, // input image width
    const int64_t kernel_height,
    const int64_t kernel_width, 
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t dilation_height,
    const int64_t dilation_width,
    const int64_t height_col, // The number of times the kernel moves along the height
    const int64_t width_col, // The number of times the kernel moves along the width
    dt* data_col) 
{
    CUDA_KERNEL_LOOP(index, n) {
        // allocate the task to (channels * kernel_height * kernel_width * height_col) num of threads
        // index = channel_in * ((kernel_height * kernel_width) * height_col) + (idx_h * kernel_width + idx_w) * height_col + h_out
        int64_t h_out = index % height_col;
        int64_t idx1 = index / height_col;
        int64_t idx_w = idx1 % kernel_width;
        int64_t idx2 = idx1 / kernel_width;
        int64_t idx_h = idx2 % kernel_height;
        int64_t channel_in = idx2 / kernel_height;

        int64_t channel_out = channel_in * kernel_height * kernel_width;
        int64_t h_in = h_out * stride_height - pad_height;
        int64_t w_in = -pad_width;

        dt* col = data_col + (height_col * width_col) * channel_out +  
                  (height_col * width_col) * (idx_h * kernel_width + idx_w) + width_col * h_out;
        const dt* im = data_im + (channel_in * height + h_in) * width + w_in;

        int64_t h = h_in + idx_h * dilation_height;
        for (int64_t i = 0; i < width_col; ++i) {
            int64_t w = w_in + i * stride_width + idx_w * dilation_width;
            *col = (h >= 0 && w >= 0 && h < height && w < width)
                ? im[idx_h * dilation_height * width + i * stride_width + idx_w * dilation_width]
                : static_cast<dt>(0);
            col ++;
        }
    }
}

// Wrapper for our_im2col kernel
template <typename dt>
void our_im2col(
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
    // We are going to launch channels * height_col * kernel_height * kernel_width kernels
    int64_t num_kernels = channels * height_col * kernel_height * kernel_width;
    printf("num_kernels of our_im2col: %d\n", num_kernels);
    int threads = 1024;
    int blocks = GET_BLOCKS(num_kernels, threads);
    // Launch CUDA_NUM_THREADS = 1024
    our_im2col_kernel<<<blocks, threads, 0, stream>>>(
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
    printf("num_kernels of origin_im2col: %d\n", num_kernels);
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
    const int64_t channels = 128;
    const int64_t height = 256;
    const int64_t width = 512;

    const int64_t kernel_height = 10;
    const int64_t kernel_width = 10;

    const int64_t pad_height = 1;
    const int64_t pad_width = 2;

    const int64_t stride_height = 2;
    const int64_t stride_width = 2;

    const int64_t dilation_height = 2;
    const int64_t dilation_width = 2;

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
    printf("Starting CUDA warm-up (%d iterations)...\n", warmup_iterations);
    for(int i = 0; i < warmup_iterations; ++i){
        // Warm-up our_im2col
        our_im2col<float>(
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
        
        // Warm-up original im2col
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

    // === Timing and Testing Phase ===
    // Create CUDA events for timing
    cudaEvent_t start_our, stop_our, start_orig, stop_orig;
    cudaEventCreate(&start_our);
    cudaEventCreate(&stop_our);
    cudaEventCreate(&start_orig);
    cudaEventCreate(&stop_orig);

    // Launch our_im2col and measure time
    cudaEventRecord(start_our, stream);
    our_im2col<float>(
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
    cudaEventRecord(stop_our, stream);

    // Launch original im2col and measure time
    cudaEventRecord(start_orig, stream);
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
    cudaEventRecord(stop_orig, stream);

    // Wait for all operations to finish
    cudaStreamSynchronize(stream);

    // Calculate elapsed time for our_im2col
    float milliseconds_our = 0;
    cudaEventElapsedTime(&milliseconds_our, start_our, stop_our);

    // Calculate elapsed time for original im2col
    float milliseconds_orig = 0;
    cudaEventElapsedTime(&milliseconds_orig, start_orig, stop_orig);

    printf("our_im2col execution time: %f ms\n", milliseconds_our);
    printf("original im2col execution time: %f ms\n", milliseconds_orig);

    // Copy results back to host
    cudaMemcpy(h_data_col_our, d_data_col_our, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data_col_orig, d_data_col_orig, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify that both outputs are the same
    bool correct = compare_arrays(h_data_col_our, h_data_col_orig, output_size);
    if (correct) {
        printf("SUCCESS: Both kernels produce the same output.\n");
    } else {
        printf("ERROR: Outputs of the kernels do not match.\n");
    }

    // Clean up
    cudaFree(d_data_im);
    cudaFree(d_data_col_our);
    cudaFree(d_data_col_orig);

    free(h_data_im);
    free(h_data_col_our);
    free(h_data_col_orig);

    cudaEventDestroy(start_our);
    cudaEventDestroy(stop_our);
    cudaEventDestroy(start_orig);
    cudaEventDestroy(stop_orig);

    cudaStreamDestroy(stream);

    return 0;
}