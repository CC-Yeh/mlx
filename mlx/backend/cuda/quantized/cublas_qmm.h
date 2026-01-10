// Copyright © 2026 Apple Inc.
#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/cublas_utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/quantization_mode.h"

#include <cublasLt.h>

namespace mlx::core {

// One-sided block-scaled matmul:
// - Activations are FP16/BF16/FP32.
// - Weights are block-scaled quantized (nvfp4/mxfp8).
// - Scales are provided separately (uint8) in block-scaled layout.
class CublasQMM : public CublasMatmulBase {
 public:
  CublasQMM(
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
      QuantizationMode quantization_mode);

  void run(
      cu::CommandEncoder& encoder,
      array& out,
      const array& a,
      const array& wq,
      const array& w_scales,
      float alpha = 1.0f);

 private:
  cublasLtMatmulMatrixScale_t w_scale_mode_;
  QuantizationMode quantization_mode_;
};

} // namespace mlx::core
