#include <torch/torch.h>
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/Functions.h>


int main() {
    // 创建输入张量 (self)、权重张量 (weight)、和输出张量 (output)
    at::Tensor self = at::zeros({1, 3, 16, 16}).cuda(); // 修改为 Channels=3
    at::Tensor weight = at::ones({2, 3, 5, 5}).cuda(); // 权重张量形状保持不变

    self[0][0][0][0] = 1;
    self[0][0][1][0] = 1;
    self[0][2][5][0] = 1;


    // 参数定义
    std::vector<int64_t> k({5,5}), st({1,1}), p({2,2});
    at::IntArrayRef kernel_size(k);  // 卷积核大小
    at::IntArrayRef stride(st);      // 步幅
    at::IntArrayRef padding(p);    // 填充
    c10::optional<at::Tensor> bias = c10::nullopt; // 无偏置

    std::cout << "kernel_size: "<< kernel_size<<std::endl;
    std::cout << "stride: "<< stride<<std::endl;
    std::cout << "padding: "<< padding<<std::endl;

    // 调用 _slow_conv2d_forward_out
    try {
        std::cout << "Input Tensor: " << self.sizes() << std::endl;
        std::cout << "Weight Tensor: " << weight.sizes() << std::endl;
        auto output = at::_our_conv2d_forward(self, weight, kernel_size, bias, stride, padding);
        std::cout << "Output Tensor: " << output.sizes() << std::endl;
        std::cout << "Output Tensor: " << output << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error in _slow_conv2d_forward_out: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}