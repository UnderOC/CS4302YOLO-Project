#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/div_rtn.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/Resize.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <ATen/native/cuda/im2col.cuh>
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_slow_conv2d_backward_native.h>
#include <ATen/ops/_slow_conv2d_forward_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/sum.h>
#endif

namespace at {
namespace native {
namespace {

void slow_conv2d_shape_check(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    const Tensor& bias,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW,
    bool weight_nullable) {
  TORCH_CHECK(
      kW > 0 && kH > 0,
      "kernel size should be greater than zero, but got kH: ",
      kH,
      " kW: ",
      kW);
  TORCH_CHECK(
      dW > 0 && dH > 0,
      "stride should be greater than zero, but got dH: ",
      dH,
      " dW: ",
      dW);

  TORCH_CHECK(
      weight_nullable || weight.defined(),
      "weight tensor is expected to be non-nullable");
  TORCH_CHECK(
      !weight.defined() || ((weight.numel() > 0) && (weight.dim() == 2)),
      "non-empty 2D weight tensor expected, but got: ",
      weight.sizes());
  TORCH_CHECK(
      !bias.defined() ||
          (bias.dim() == 1 && bias.sizes()[0] == weight.sizes()[0]),
      "Expected bias to have shape [",
      weight.sizes()[0],
      "] but got ",
      bias.sizes());

  const auto in_sizes = input.sizes();
  constexpr int ndim = 4;
  constexpr int dimf = 1;
  constexpr int dimh = 2;
  constexpr int dimw = 3;
  TORCH_CHECK(
      in_sizes.size() == ndim, "Expected 4D input tensor, but got ", in_sizes);

  // Allow for empty batch size but not other dimensions
  const bool valid_empty = c10::multiply_integers(in_sizes.slice(1)) != 0;
  TORCH_CHECK(
      valid_empty, "non-empty input tensor expected but got: ", in_sizes);

  int64_t inputHeight = in_sizes[dimh];
  int64_t inputWidth = in_sizes[dimw];

  int64_t exactInputHeight = inputHeight + 2 * padH;
  int64_t exactInputWidth = inputWidth + 2 * padW;

  TORCH_CHECK(
      exactInputHeight >= kH && exactInputWidth >= kW,
      "Calculated padded input size per channel: ",
      IntArrayRef{exactInputHeight, exactInputWidth},
      ". Kernel size: ",
      IntArrayRef{kH, kW},
      ". Kernel size can't be greater than actual input size");

  // NOTE: can't use conv_output_size if the weight isn't defined
  auto outputHeight = div_rtn<int64_t>(exactInputHeight - kH, dH) + 1;
  auto outputWidth = div_rtn<int64_t>(exactInputWidth - kW, dW) + 1;

  TORCH_CHECK(
      outputWidth >= 1 && outputHeight >= 1,
      "Given input size per channel: ",
      IntArrayRef{inputHeight, inputWidth},
      ". Calculated output size per channel: ",
      IntArrayRef{outputHeight, outputWidth},
      ". Output size is too small");

  if (weight.defined()) {
    const auto w_sizes = weight.sizes();
    int64_t nInputPlane = w_sizes[1];
    if (w_sizes.size() == 2) {
      nInputPlane /= (kH * kW);
    }
    TORCH_CHECK(
        in_sizes[dimf] == nInputPlane,
        "Expected input dim ",
        dimf,
        " to have size ",
        nInputPlane,
        " but got ",
        in_sizes[dimf]);
  }

  if (grad_output.defined()) {
    const auto gO_sizes = grad_output.sizes();
    TORCH_CHECK(
        gO_sizes.size() == ndim,
        "Expected grad_output to have ",
        ndim,
        " dimensions but got shape",
        gO_sizes);

    if (weight.defined()) {
      const auto w_sizes = weight.sizes();
      TORCH_CHECK(
          gO_sizes[dimf] == w_sizes[0],
          "Expected  dim ",
          dimf,
          " to have size ",
          w_sizes[0],
          " but got ",
          gO_sizes[dimf]);
    } else if (bias.defined()) {
      const auto b_sizes = bias.sizes();
      int64_t nOutputPlane = b_sizes.size() == 0 ? 1 : b_sizes[0];
      TORCH_CHECK(
          gO_sizes[dimf] == nOutputPlane,
          "Expected grad_output dim ",
          dimf,
          " to have size ",
          nOutputPlane,
          " but got ",
          gO_sizes[dimf]);
    }
    TORCH_CHECK(
        gO_sizes[dimh] == outputHeight,
        "Expected grad_output dim ",
        dimh,
        " to have size ",
        outputHeight,
        " but got ",
        gO_sizes[dimh]);
    TORCH_CHECK(
        gO_sizes[dimw] == outputWidth,
        "Expected grad_output dim ",
        dimw,
        " to have size ",
        outputWidth,
        " but got ",
        gO_sizes[dimw]);
  }
}

#define BLOCKSIZE 16
template <typename T>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void MyGemmKernel(T* A, T* B, T* C, int m, int n, int k) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y * BLOCKSIZE + ty * 2;
  int col = blockIdx.x * BLOCKSIZE + tx * 2;
  float2 r1, r2, r3, r4;
  T sum[2][2] = {0};

  __shared__ __align__(16 * 512) T As[BLOCKSIZE][BLOCKSIZE];
  __shared__ __align__(16 * 512) T Bs[BLOCKSIZE][BLOCKSIZE];

  bool loadRowA = row < m && tx * 2 < k;
  bool loadRowA1 = row + 1 < m && tx * 2 < k;
  bool loadColB = col < n && ty * 2 < k;
  bool loadColB1 = col + 1 < n && ty * 2 < k;

  reinterpret_cast<float2*>(As[ty * 2])[tx] = loadRowA
      ? reinterpret_cast<float2*>(A + row * k)[tx]
      : make_float2(0.0, 0.0);
  reinterpret_cast<float2*>(As[ty * 2 + 1])[tx] = loadRowA1
      ? reinterpret_cast<float2*>(A + (row + 1) * k)[tx]
      : make_float2(0.0, 0.0);

  reinterpret_cast<float2*>(Bs[ty * 2])[tx] = loadColB
      ? reinterpret_cast<float2*>(B + (ty * 2) * n)[col / 2]
      : make_float2(0.0, 0.0);
  reinterpret_cast<float2*>(Bs[ty * 2 + 1])[tx] = loadColB1
      ? reinterpret_cast<float2*>(
            B + (ty * 2 + 1) * n)[col / 2]
      : make_float2(0.0, 0.0);

  for (int t = 1; t < (k - 1) / BLOCKSIZE + 1; t++) {

    __syncthreads();

// 展开外部循环
#pragma unroll
    for (int i = 0; i < BLOCKSIZE; ++i) {
      sum[0][0] = fma(As[ty * 2][i], Bs[i][tx * 2], sum[0][0]);
      sum[0][1] = fma(As[ty * 2][i], Bs[i][tx * 2 + 1], sum[0][1]);
      sum[1][0] = fma(As[ty * 2 + 1][i], Bs[i][tx * 2], sum[1][0]);
      sum[1][1] = fma(As[ty * 2 + 1][i], Bs[i][tx * 2 + 1], sum[1][1]);
    }

    loadRowA = row < m && t * BLOCKSIZE + tx * 2 < k;
    loadRowA1 = row + 1 < m && t* BLOCKSIZE + tx * 2 < k;
    loadColB = col < n && t * BLOCKSIZE + ty * 2 < k;
    loadColB1 = col + 1 < n && t * BLOCKSIZE + ty * 2 < k;

    r1 = loadRowA
        ? reinterpret_cast<float2*>(A + row * k + t * BLOCKSIZE)[tx]
        : make_float2(0.0, 0.0);
    r2 = loadRowA1
        ? reinterpret_cast<float2*>(A + (row + 1) * k + t * BLOCKSIZE)[tx]
        : make_float2(0.0, 0.0);

    r3 = loadColB
        ? reinterpret_cast<float2*>(B + (t * BLOCKSIZE + ty * 2) * n)[col / 2]
        : make_float2(0.0, 0.0);
    r4 = loadColB1
        ? reinterpret_cast<float2*>(
              B + (t * BLOCKSIZE + ty * 2 + 1) * n)[col / 2]
        : make_float2(0.0, 0.0);
    

    __syncthreads();

    reinterpret_cast<float2*>(As[ty * 2])[tx] = r1;
    reinterpret_cast<float2*>(As[ty * 2 + 1])[tx] = r2;

    reinterpret_cast<float2*>(Bs[ty * 2])[tx] = r3;
    reinterpret_cast<float2*>(Bs[ty * 2 + 1])[tx] = r4;
  }

  __syncthreads();

// 展开外部循环
#pragma unroll
  for (int i = 0; i < BLOCKSIZE; ++i) {
    sum[0][0] = fma(As[ty * 2][i], Bs[i][tx * 2], sum[0][0]);
    sum[0][1] = fma(As[ty * 2][i], Bs[i][tx * 2 + 1], sum[0][1]);
    sum[1][0] = fma(As[ty * 2 + 1][i], Bs[i][tx * 2], sum[1][0]);
    sum[1][1] = fma(As[ty * 2 + 1][i], Bs[i][tx * 2 + 1], sum[1][1]);
  }

  __syncthreads();

  bool rowInRange = row < m;
  bool nextRowInRange = (row + 1) < m;
  bool colInRange = col < n;
  bool nextColInRange = (col + 1) < n;
  if (rowInRange && colInRange)
    C[row * n + col] += sum[0][0];
  if (rowInRange && nextColInRange)
    C[row * n + col + 1] += sum[0][1];
  if (nextRowInRange && colInRange)
    C[(row + 1) * n + col] += sum[1][0];
  if (nextRowInRange && nextColInRange)
    C[(row + 1) * n + col + 1] += sum[1][1];

}

template <typename T>
void MyGemm(cudaStream_t stream, T* A, T* B, T* C, int m, int n, int k) {
  dim3 dimBlock(BLOCKSIZE / 2, BLOCKSIZE / 2);
  dim3 dimGrid(
      (n + BLOCKSIZE - 1) / (BLOCKSIZE), (m + BLOCKSIZE - 1) / (BLOCKSIZE));
  MyGemmKernel<<<dimGrid, dimBlock, 0, stream>>>(A, B, C, m, n, k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

Tensor new_view_weight_MM2d(const Tensor& weight_) {
  auto weight = weight_.expect_contiguous();
  const auto w_sizes = weight->sizes();
  TORCH_CHECK(w_sizes.size() == 4);
  int64_t s1 = w_sizes[0];
  int64_t s2 = c10::multiply_integers(w_sizes.slice(1));
  return weight->view({s1, s2});
}

template <typename dt>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void my_im2col_kernel(
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
    dt* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int64_t h_out = index % height_col;
    int64_t idx1 = index / height_col;
    int64_t w_kernel = idx1 % kernel_width;
    int64_t idx2 = idx1 / kernel_width;
    int64_t h_kernel = idx2 % kernel_height;
    int64_t channel_in = idx2 / kernel_height;

    int64_t channel_out = channel_in * kernel_height * kernel_width;
    int64_t h_in = h_out * stride_height - pad_height;
    int64_t w_in = -pad_width;

    dt* col = data_col + (channel_out * height_col + h_out) * width_col +
        height_col * width_col * (h_kernel * kernel_width + w_kernel);
    const dt* im = data_im + (channel_in * height + h_in) * width + w_in;

    int64_t h = h_in + h_kernel * dilation_height;
    for (int64_t i = 0; i < width_col; ++i) {
      int64_t w = w_in + i * stride_width + w_kernel * dilation_width;
      *col = (h >= 0 && w >= 0 && h < height && w < width)
          ? im[h_kernel * dilation_height * width + i * stride_width +
               w_kernel * dilation_width]
          : static_cast<dt>(0);
      col++;
    }
  }
}

template <typename dt>
void my_my_im2col(
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
    dt* data_col) {
  // We are going to launch channels * height_col * kernel_height * kernel_width, 
  // each kernel responsible for copying a single-channel grid.
  int64_t num_kernels = channels * height_col * kernel_height * kernel_width;
  // Launch CUDA_NUM_THREADS = 1024, the same as the default version.
  my_im2col_kernel<<<GET_BLOCKS(num_kernels), 1024, 0, stream>>>(
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

template <typename T>
bool verification(T* data1, T* data2, int elementsNum, double atol = 5e-5) {
  double res = 0;
  for (int i = 0; i < elementsNum; i++) {
    double value1 = static_cast<double>(data1[i]);
    double value2 = static_cast<double>(data2[i]);
    double tmp = abs(value1 - value2);
    if (tmp > res) {
      res = tmp;
    }
  }
  printf("\nThe max precision error is :%f", res);
  if (res > atol)
    return false;
  else
    return true;
}

void slow_conv2d_forward(
    const Tensor& input,
    const Tensor& output,
    const Tensor& weight_,
    const Tensor& bias,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW) {
  auto weight = new_view_weight_MM2d(weight_);
  //   printf("\n slow_conv2d. Start!\n");
  //   printf("\n The Infomation About Input");
  //   printf("\tDimension: %" PRId64 "\n", input.dim());
  //   for (int i = 0; i < input.dim(); i++) {
  //     printf("\tSize: %" PRId64 "\n", input.size(i));
  //   }
  //   printf("\n The Infomation About weight_");
  //   printf("\tDimension: %" PRId64 "\n", weight_.dim());
  //   for (int i = 0; i < weight_.dim(); i++) {
  //     printf("\tSize: %" PRId64 "\n", weight_.size(i));
  //   }

  slow_conv2d_shape_check(
      input,
      {},
      weight,
      bias,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      /*weight_nullable*/ false);

  constexpr int dimf = 1;
  constexpr int dimh = 2;
  constexpr int dimw = 3;

  auto in_sizes = input.sizes();
  int64_t batchSize = in_sizes[0];
  int64_t nInputPlane = in_sizes[dimf];
  int64_t inputHeight = in_sizes[dimh];
  int64_t inputWidth = in_sizes[dimw];
  int64_t nOutputPlane = weight.sizes()[0];
  int64_t outputHeight = (inputHeight + 2 * padH - kH) / dH + 1;
  int64_t outputWidth = (inputWidth + 2 * padW - kW) / dW + 1;

  // Resize output
  resize_output(output, {batchSize, nOutputPlane, outputHeight, outputWidth});

  // Create temporary columns
  auto columns = at::empty(
      {nInputPlane * kW * kH, outputHeight * outputWidth}, input.options());

  // For verification
  // *****************************************************************
  auto columns_standard = at::empty(
      {nInputPlane * kW * kH, outputHeight * outputWidth}, input.options());

  const bool requires_columns =
      (kW != 1 || kH != 1 || dW != 1 || dH != 1 || padH != 0 || padW != 0);

  if (bias.defined()) {
    TORCH_CHECK(
        bias.scalar_type() == input.scalar_type(),
        "Expected bias to have type ",
        input.scalar_type(),
        " but got ",
        bias.scalar_type());
    output.copy_(bias.view({-1, 1, 1}));
  } else {
    output.zero_();
  }
  float GemmTime = 0;
  float im2colTime_default = 0;
  float im2colTime = 0;
  float milliseconds = 0;
  float MyGemmTime = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, input.scalar_type(), "slow_conv2d_cuda", [&] {
        // For each elt in batch, do:
        for (int elt = 0; elt < batchSize; elt++) {
          auto input_n = input.select(0, elt);
          auto output_n = output.select(0, elt);
          auto myOutput = output_n.clone();
          if (requires_columns) {
            // Extract columns:
            //************************************************************************************************************

            /*cudaEventRecord(start, c10::cuda::getCurrentCUDAStream());
            cudaEventSynchronize(start);
            at::native::my_im2col(
                c10::cuda::getCurrentCUDAStream(),
                input_n.data_ptr<scalar_t>(),
                nInputPlane,
                inputHeight,
                inputWidth,
                outputHeight,
                outputWidth,
                kH,
                kW,
                padH,
                padW,
                dH,
                dW,
                1,
                1,
                columns_standard.data_ptr<scalar_t>());
            cudaEventRecord(stop, c10::cuda::getCurrentCUDAStream());
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            im2colTime_default += milliseconds;*/

            cudaEventRecord(start, c10::cuda::getCurrentCUDAStream());
            cudaEventSynchronize(start);
            my_my_im2col(
                c10::cuda::getCurrentCUDAStream(),
                input_n.data_ptr<scalar_t>(),
                nInputPlane,
                inputHeight,
                inputWidth,
                outputHeight,
                outputWidth,
                kH,
                kW,
                padH,
                padW,
                dH,
                dW,
                1,
                1,
                columns.data_ptr<scalar_t>());
            cudaEventRecord(stop, c10::cuda::getCurrentCUDAStream());
            cudaEventSynchronize(stop);
            milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            im2colTime += milliseconds;

            // printf(
            //     "\n%" PRId64 " %" PRId64 " %" PRId64 " %" PRId64 " %" PRId64
            //     "", nInputPlane, outputHeight, outputWidth, kH, kW);
            /*printf("\n|- Start Verfication.");
            if (columns.device().is_cuda() &&
                columns_standard.device().is_cuda())
              printf("\n |- Data of Im2Col is all in Cuda");
            else
              printf("\n |- Data of Im2Col is not all in Cuda");
            Tensor columns_standard_cpu =
                columns_standard.to(at::Device(at::kCPU));
            Tensor columns_cpu = columns.to(at::Device(at::kCPU));
            if (columns_standard.numel() == columns.numel() &&
                verification<scalar_t>(
                    columns_standard_cpu.data_ptr<scalar_t>(),
                    columns_cpu.data_ptr<scalar_t>(),
                    columns.numel()))
              printf("\n |- Verifying My Im2Col.....YES!\n");
            else {
              printf("\n |- Verification Fail: Unexpected Precision Error!");
              exit(-1);
            }*/
          }

          // M,N,K are dims of matrix A and B
          // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
          int64_t m = nOutputPlane;
          int64_t n = columns.size(1);
          int64_t k = nInputPlane * kH * kW;

          // Do GEMM (note: this is a bit confusing because gemm assumes
          // column-major matrices)
          //*************************************************************************************************************
          auto gemm_in_ptr = requires_columns ? columns_standard.data_ptr<scalar_t>()
                                              : input_n.data_ptr<scalar_t>();

          cudaEventRecord(start, c10::cuda::getCurrentCUDAStream());
          cudaEventSynchronize(start);
          MyGemm(
              c10::cuda::getCurrentCUDAStream(),
              weight.data_ptr<scalar_t>(),
              columns.data_ptr<scalar_t>(),
              myOutput.data_ptr<scalar_t>(),
              m,
              n,
              k);
          cudaEventRecord(stop, c10::cuda::getCurrentCUDAStream());
          cudaEventSynchronize(stop);
          milliseconds = 0;
          cudaEventElapsedTime(&milliseconds, start, stop);
          MyGemmTime += milliseconds;

          /*cudaEventRecord(start, c10::cuda::getCurrentCUDAStream());
          cudaEventSynchronize(start);
          at::cuda::blas::gemm(
              'n',
              'n',
              n,
              m,
              k,
              scalar_t(1),
              gemm_in_ptr,
              n,
              weight.data_ptr<scalar_t>(),
              k,
              scalar_t(1),
              output_n.data_ptr<scalar_t>(),
              n);
          cudaEventRecord(stop, c10::cuda::getCurrentCUDAStream());
          cudaEventSynchronize(stop);
          milliseconds = 0;
          cudaEventElapsedTime(&milliseconds, start, stop);
          GemmTime += milliseconds;*/

          /*Tensor myGemmRes = myOutput.to(at::Device(at::kCPU));
          Tensor GemmRes = output_n.to(at::Device(at::kCPU));
          if (myGemmRes.sizes() != GemmRes.sizes()) {
            printf("\nUnexpected size in output tensor of myGemm");
          }
          if (myOutput.numel() == output_n.numel() &&
              verification<scalar_t>(
                  myGemmRes.data_ptr<scalar_t>(),
                  GemmRes.data_ptr<scalar_t>(),
                  myOutput.numel()))
            printf("\nVerifying My Gemm.....YES!\n");
          else {
            printf("\nVerification Fail: Unexpected Precision Error!");
            exit(-1);
          }
          printf("\n|- End Verfication");*/
        }
        //printf("\nIm2Col Time: %f", im2colTime_default);
        printf("\nMyIm2Col Time: %f", im2colTime);
        //printf("\nGEMM Time: %f", GemmTime);
        printf("\nMyGEMM Time: %f", MyGemmTime);
        //printf("\nAccleration Rate for Im2Col is : %.3f", im2colTime_default / im2colTime);
        //printf("\nAccleration Rate for GEMM is : %.3f", GemmTime / MyGemmTime);
        printf("\n");
      });
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

void slow_conv2d_backward(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& grad_input,
    const Tensor& weight_,
    const Tensor& grad_columns,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW) {
  Tensor weight = new_view_weight_MM2d(weight_);
  slow_conv2d_shape_check(
      input,
      grad_output,
      weight,
      {},
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      /*weight_nullable=*/false);

  // Params
  auto weight_sizes = weight.sizes();
  int nInputPlane = weight_sizes[1] / (kW * kH);
  int nOutputPlane = weight_sizes[0];

  TORCH_INTERNAL_ASSERT(grad_output.is_contiguous());

  auto input_sizes = input.sizes();
  int64_t inputWidth = input_sizes[3];
  int64_t inputHeight = input_sizes[2];
  auto output_sizes = grad_output.sizes();
  int64_t outputWidth = output_sizes[3];
  int64_t outputHeight = output_sizes[2];

  // Batch size + input planes
  int64_t batchSize = input_sizes[0];

  // Resize output
  resize_output(grad_input, input_sizes);
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");

  // Resize temporary columns
  resize_output(
      grad_columns, {nInputPlane * kW * kH, outputHeight * outputWidth});
  TORCH_CHECK(grad_columns.is_contiguous(), "grad_columns must be contiguous");

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, input.scalar_type(), "slow_conv2d_backward_cuda", [&] {
        // For each elt in batch, do:
        for (int elt = 0; elt < batchSize; elt++) {
          // Matrix mulitply per sample:
          auto grad_input_n = grad_input.select(0, elt);
          auto grad_output_n = grad_output.select(0, elt);

          // M,N,K are dims of matrix A and B
          // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
          int64_t m = nInputPlane * kW * kH;
          int64_t n = grad_columns.sizes()[1];
          int64_t k = nOutputPlane;

          // Do GEMM (note: this is a bit confusing because gemm assumes
          // column-major matrices)
          at::cuda::blas::gemm<scalar_t>(
              'n',
              't',
              n,
              m,
              k,
              scalar_t(1),
              grad_output_n.data_ptr<scalar_t>(),
              n,
              weight.data_ptr<scalar_t>(),
              m,
              scalar_t(0),
              grad_columns.data_ptr<scalar_t>(),
              n);

          // Unpack columns back into input:
          using acc_t = at::acc_type<scalar_t, true>;
          at::native::col2im<scalar_t, acc_t>(
              c10::cuda::getCurrentCUDAStream(),
              grad_columns.data_ptr<scalar_t>(),
              nInputPlane,
              inputHeight,
              inputWidth,
              outputHeight,
              outputWidth,
              kH,
              kW,
              padH,
              padW,
              dH,
              dW,
              1,
              1,
              grad_input_n.data_ptr<scalar_t>());
        }
      });
}

void slow_conv2d_grad_weight(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& grad_weight_,
    const Tensor& columns,
    int64_t kH,
    int64_t kW,
    int64_t dH,
    int64_t dW,
    int64_t padH,
    int64_t padW) {
  TORCH_CHECK(
      grad_weight_.is_contiguous(), "grad_weight needs to be contiguous");
  auto grad_weight = new_view_weight_MM2d(grad_weight_);
  slow_conv2d_shape_check(
      input,
      grad_output,
      grad_weight,
      {},
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      /*weight_nullable=*/true);

  // Params
  TORCH_INTERNAL_ASSERT(input.is_contiguous());
  TORCH_INTERNAL_ASSERT(grad_output.is_contiguous());

  auto input_sizes = input.sizes();
  int64_t nInputPlane = input_sizes[1];
  int64_t nOutputPlane = grad_output.sizes()[1];

  int64_t inputWidth = input_sizes[3];
  int64_t inputHeight = input_sizes[2];
  int64_t outputWidth = (inputWidth + 2 * padW - kW) / dW + 1;
  int64_t outputHeight = (inputHeight + 2 * padH - kH) / dH + 1;

  // Batch size + input planes
  int64_t batchSize = input_sizes[0];

  // Resize temporary columns
  resize_output(columns, {nInputPlane * kH * kW, outputHeight * outputWidth});

  const bool requires_columns =
      (kW != 1 || kH != 1 || dW != 1 || dH != 1 || padH != 0 || padW != 0);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      input.scalar_type(),
      "slow_conv2d_grad_weight_cuda",
      [&] {
        // For each elt in batch, do:
        for (int elt = 0; elt < batchSize; elt++) {
          // Matrix mulitply per output:
          auto grad_output_n = grad_output.select(0, elt);

          // Matrix mulitply per output:
          auto input_n = input.select(0, elt);

          if (requires_columns) {
            // Extract columns:
            at::native::im2col<scalar_t>(
                c10::cuda::getCurrentCUDAStream(),
                input_n.data_ptr<scalar_t>(),
                nInputPlane,
                inputHeight,
                inputWidth,
                outputHeight,
                outputWidth,
                kH,
                kW,
                padH,
                padW,
                dH,
                dW,
                1,
                1,
                columns.data_ptr<scalar_t>());
          }

          // M,N,K are dims of matrix A and B
          // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
          int64_t m = nOutputPlane;
          int64_t n = nInputPlane * kW * kH;
          int64_t k = columns.sizes()[1];

          // Do GEMM (note: this is a bit confusing because gemm assumes
          // column-major matrices)
          auto gemm_in_ptr = requires_columns ? columns.data_ptr<scalar_t>()
                                              : input_n.data_ptr<scalar_t>();
          at::cuda::blas::gemm(
              't',
              'n',
              n,
              m,
              k,
              scalar_t(1),
              gemm_in_ptr,
              k,
              grad_output_n.data_ptr<scalar_t>(),
              k,
              scalar_t(1),
              grad_weight.data_ptr<scalar_t>(),
              n);
        }
      });
}

} // namespace

Tensor& slow_conv2d_forward_out_cuda(
    const Tensor& self_,
    const Tensor& weight_,
    IntArrayRef kernel_size,
    const c10::optional<Tensor>& bias_,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
  TORCH_CHECK(kernel_size.size() == 2);
  TORCH_CHECK(stride.size() == 2);
  TORCH_CHECK(padding.size() == 2);

  auto self = self_.expect_contiguous();
  auto weight = weight_.expect_contiguous();
  auto bias = [&] {
    if (bias_.has_value() && bias_->defined()) {
      return bias_->expect_contiguous();
    }
    return MaybeOwned<Tensor>::owned(c10::in_place);
  }();

  slow_conv2d_forward(
      *self,
      output,
      *weight,
      *bias,
      kernel_size[0],
      kernel_size[1],
      stride[0],
      stride[1],
      padding[0],
      padding[1]);
  return output;
}

Tensor slow_conv2d_forward_cuda(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const c10::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding) {
  auto output = at::empty({0}, self.options());
  return slow_conv2d_forward_out_cuda(
      self, weight, kernel_size, bias, stride, padding, output);
}

std::tuple<Tensor&, Tensor&, Tensor&> slow_conv2d_backward_out_cuda(
    const Tensor& grad_output_,
    const Tensor& self_,
    const Tensor& weight_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias) {
  auto grad_output = grad_output_.expect_contiguous();

  Tensor columns = at::empty({0}, self_.options());
  if (grad_input.defined()) {
    resize_output(grad_input, self_.sizes());
    auto weight = weight_.expect_contiguous();

    slow_conv2d_backward(
        self_,
        *grad_output,
        grad_input,
        *weight,
        columns,
        kernel_size[0],
        kernel_size[1],
        stride[0],
        stride[1],
        padding[0],
        padding[1]);
  }
  if (grad_bias.defined()) {
    at::sum_out(grad_bias, *grad_output, IntArrayRef{0, 2, 3});
  }
  if (grad_weight.defined()) {
    resize_output(grad_weight, weight_.sizes());
    grad_weight.zero_();
    auto self = self_.expect_contiguous();
    slow_conv2d_grad_weight(
        *self,
        *grad_output,
        grad_weight,
        columns,
        kernel_size[0],
        kernel_size[1],
        stride[0],
        stride[1],
        padding[0],
        padding[1]);
  }
  return std::tuple<Tensor&, Tensor&, Tensor&>{
      grad_input, grad_weight, grad_bias};
}

std::tuple<Tensor, Tensor, Tensor> slow_conv2d_backward_cuda(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    std::array<bool, 3> output_mask) {
  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;

  if (output_mask[0]) {
    grad_input = at::empty({0}, grad_output.options());
  }

  if (output_mask[1]) {
    grad_weight = at::empty({0}, grad_output.options());
  }

  if (output_mask[2]) {
    grad_bias = at::empty({0}, grad_output.options());
  }

  return native::slow_conv2d_backward_out_cuda(
      grad_output,
      self,
      weight,
      kernel_size,
      stride,
      padding,
      grad_input,
      grad_weight,
      grad_bias);
}

} // namespace native
} // namespace at