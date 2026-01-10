// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/quantized/cublas_qmm.h"
#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

// TODO: reuse existing helper from quantized.cpp. refactor out. should be
// backend agnostic

inline array ensure_row_contiguous(
    const array& x,
    cu::CommandEncoder& enc,
    const Stream& s) {
  if (!x.flags().row_contiguous) {
    array x_copy = contiguous_copy_gpu(x, s);
    enc.add_temporary(x_copy);
    return x_copy;
  } else {
    return x;
  }
}

inline array ensure_row_contiguous_matrix(
    const array& x,
    cu::CommandEncoder& enc,
    const Stream& s) {
  auto stride_0 = x.strides()[x.ndim() - 2];
  auto stride_1 = x.strides()[x.ndim() - 1];
  if (stride_0 == x.shape(-1) && stride_1 == 1) {
    return x;
  }
  array x_copy = contiguous_copy_gpu(x, s);
  enc.add_temporary(x_copy);
  return x_copy;
}

} // namespace

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& enc = cu::get_command_encoder(s);
  auto& device = enc.device();
  // TODO: refactor to utils
  auto cc = device.compute_capability_major() * 100 +
      device.compute_capability_minor() * 10;
  if (cc < 1000) {
    throw std::runtime_error(
        "[QuantizedMatmul::eval_gpu] is only supported on GPUs with compute capability 10.0 or higher.");
  }

  auto x = ensure_row_contiguous_matrix(inputs[0], enc, s);
  auto wq = ensure_row_contiguous_matrix(inputs[1], enc, s);
  auto scales = ensure_row_contiguous_matrix(inputs[2], enc, s);

  // TODO, support qmv and batch
  // Current only handles 2D inputs and block-scaled modes.
  if (x.ndim() != 2 || wq.ndim() != 2) {
    throw std::runtime_error(
        "[QuantizedMatmul::eval_gpu] Only 2D inputs supported on CUDA path (yet).");
  }
  if (mode_ != QuantizationMode::Nvfp4 && mode_ != QuantizationMode::Mxfp8) {
    throw std::runtime_error(
        "[QuantizedMatmul::eval_gpu] CUDA path only implemented for nvfp4/mxfp8.");
  }

  out.set_data(cu::malloc_async(out.nbytes(), enc));

  int K = x.shape(-1);
  int M = x.shape(-2);
  int N = out.shape(-1);
  bool x_transposed = false;
  bool w_transposed = transpose_;
  int64_t lda = K;
  int64_t ldb = K;

  CublasQMM qmm(
      enc.device(),
      x_transposed,
      M,
      K,
      lda,
      w_transposed,
      K,
      N,
      ldb,
      /*batch_count=*/1,
      /*a_batch_stride=*/0,
      /*b_batch_stride=*/0,
      out.dtype(),
      mode_);
  qmm.run(enc, out, x, wq, scales, /*alpha=*/1.0f);
}

} // namespace mlx::core
