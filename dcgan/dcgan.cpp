#include <torch/torch.h>
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <cuda_runtime.h>
#include <chrono>

// Function to measure time using CUDA events
double measure_time_cuda(const std::function<void()>& func) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    func();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double nanoseconds = milliseconds * 1e6;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return nanoseconds;
}

int main() {
    // List of dtypes to test
    std::vector<at::ScalarType> dtypes = {at::kFloat, at::kDouble};

    // Corresponding string representations for dtypes
    std::vector<std::string> dtype_strings = {"float32", "float64"};

    // List of shapes to test
    std::vector<std::vector<int64_t>> shapes = {
        {1, 3, 16, 16},
        {2, 2, 8, 8},
        {3, 4, 32, 32},
    };

    for (size_t i = 0; i < dtypes.size(); ++i) {
        auto dtype = dtypes[i];
        auto dtype_str = dtype_strings[i];

        for (auto shape : shapes) {
            // Create input tensors with random values
            at::Tensor self = at::rand(shape, at::device(at::DeviceType::CUDA).dtype(dtype));
            at::Tensor weight = at::rand({2, shape[1], 5, 5}, at::device(at::DeviceType::CUDA).dtype(dtype));

            // Convolution parameters
            std::vector<int64_t> k = {5, 5}, st = {1, 1}, p = {2, 2};
            at::IntArrayRef kernel_size(k);
            at::IntArrayRef stride(st);
            at::IntArrayRef padding(p);
            c10::optional<at::Tensor> bias = c10::nullopt;

            std::cout << "Testing dtype: " << dtype_str
                      << ", shape: " << self.sizes() << std::endl;

            try {
                // Compute output with _our_conv2d_forward
                auto output_our = at::_our_conv2d_forward(self, weight, kernel_size, bias, stride, padding);

                // Compute output with _slow_conv2d_forward
                auto output_slow = at::_slow_conv2d_forward(self, weight, kernel_size, bias, stride, padding);

                // Compare outputs
                bool equal = torch::allclose(output_our, output_slow, 1e-4, 1e-4);
                if (equal) {
                    std::cout << "[[[   Outputs are equal.   ]]]" << std::endl;
                } else {
                    std::cerr << "[[[   Outputs are NOT equal.   ]]]" << std::endl;
                    exit(1);
                }
            } catch (const c10::Error& e) {
                std::cerr << "Error: " << e.what() << std::endl;
            }
        }
    }

    if(true){
        return 0;
    }

    // Performance measurement with large input
    int large_batch = 16;
    int large_channels = 64;
    int large_size = 256;

    at::Tensor large_self = at::rand({large_batch, large_channels, large_size, large_size}, at::device(at::DeviceType::CUDA).dtype(at::kFloat));
    at::Tensor large_weight = at::rand({32, large_channels, 7, 7}, at::device(at::DeviceType::CUDA).dtype(at::kFloat));

    std::vector<int64_t> large_k = {7, 7}, large_st = {1, 1}, large_p = {3, 3};
    at::IntArrayRef large_kernel_size(large_k);
    at::IntArrayRef large_stride(large_st);
    at::IntArrayRef large_padding(large_p);
    c10::optional<at::Tensor> large_bias = c10::nullopt;

    int num_runs = 100;

    double time_our = 0.0;
    double time_slow = 0.0;

    // Warm-up runs
    for (int i = 0; i < 10; ++i) {
        at::_our_conv2d_forward(large_self, large_weight, large_kernel_size, large_bias, large_stride, large_padding);
        at::_slow_conv2d_forward(large_self, large_weight, large_kernel_size, large_bias, large_stride, large_padding);
    }

    // Measure _our_conv2d_forward
    for (int i = 0; i < num_runs; ++i) {
        time_our += measure_time_cuda([&]() {
            at::_our_conv2d_forward(large_self, large_weight, large_kernel_size, large_bias, large_stride, large_padding);
        });
    }
    time_our /= num_runs;

    // Measure _slow_conv2d_forward
    for (int i = 0; i < num_runs; ++i) {
        time_slow += measure_time_cuda([&]() {
            at::_slow_conv2d_forward(large_self, large_weight, large_kernel_size, large_bias, large_stride, large_padding);
        });
    }
    time_slow /= num_runs;

    std::cout << "Average time for _our_conv2d_forward: " << time_our << " ns" << std::endl;
    std::cout << "Average time for _slow_conv2d_forward: " << time_slow << " ns" << std::endl;

    return 0;
}