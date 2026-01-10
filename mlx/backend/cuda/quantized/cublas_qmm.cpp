// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/quantized/cublas_qmm.h"

#include <fmt/format.h>

namespace mlx::core {

namespace {

cudaDataType_t qmode_to_cublas_weight_dtype(QuantizationMode mode) {
  if (mode == QuantizationMode::Mxfp8) {
    return CUDA_R_8F_E4M3;
  } else if (mode == QuantizationMode::Nvfp4) {
    return CUDA_R_4F_E2M1;
  } else {
    throw std::runtime_error(fmt::format(
        "Unsupported quantization mode in CublasQMM: {}.",
        quantization_mode_to_string(mode)));
  }
}

cublasLtMatmulMatrixScale_t qmode_to_cublas_scale_mode(QuantizationMode mode) {
  if (mode == QuantizationMode::Mxfp8) {
    return CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  } else if (mode == QuantizationMode::Nvfp4) {
    return CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  } else {
    throw std::runtime_error(fmt::format(
        "Unsupported quantization mode in CublasQMM: {}.",
        quantization_mode_to_string(mode)));
  }
}

} // namespace

CublasQMM::CublasQMM(
    cu::Device& device,
    bool a_transposed,
    uint64_t a_rows,
    uint64_t a_cols,
    int64_t lda,
    bool b_transposed,
    uint64_t b_rows,
    uint64_t b_cols,
    int64_t ldb,
    int32_t batch_count,
    int64_t a_batch_stride,
    int64_t b_batch_stride,
    Dtype out_dtype,
    QuantizationMode qmode)
    : quantization_mode_(qmode) {
  cudaDataType_t scale_type = CUDA_R_32F;
  cublasComputeType_t gemm_compute_type = CUBLAS_COMPUTE_32F;
  cudaDataType_t output_type =
      cublas_utils::dtype_to_cublas_type(out_dtype, "CublasQMM");
  cudaDataType_t weight_type = qmode_to_cublas_weight_dtype(quantization_mode_);

  init_base(
      device,
      scale_type,
      gemm_compute_type,
      weight_type,
      output_type,
      a_transposed,
      a_rows,
      a_cols,
      lda,
      b_transposed,
      b_rows,
      b_cols,
      ldb,
      batch_count,
      a_batch_stride,
      b_batch_stride);

  w_scale_mode_ = qmode_to_cublas_scale_mode(quantization_mode_);
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
      &w_scale_mode_,
      sizeof(w_scale_mode_)));
}

void CublasQMM::run(
    cu::CommandEncoder& encoder,
    array& out,
    const array& a,
    const array& wq,
    const array& w_scales,
    float alpha /* = 1.0f */) {
  encoder.set_input_array(a);
  encoder.set_input_array(wq);
  encoder.set_input_array(w_scales);
  encoder.set_output_array(out);

  auto* scale_ptr = gpu_ptr<void>(w_scales);
  CHECK_CUBLAS_ERROR(cublasLtMatmulDescSetAttribute(
      matmul_desc_,
      CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
      &scale_ptr,
      sizeof(scale_ptr)));

  const void* alpha_ptr = &alpha;
  const float beta = 0.0f;
  const void* beta_ptr = &beta;

  execute_matmul(
      encoder,
      gpu_ptr<void>(out),
      gpu_ptr<void>(a),
      gpu_ptr<void>(wq),
      /*c=*/nullptr,
      alpha_ptr,
      beta_ptr);
}

} // namespace mlx::core
