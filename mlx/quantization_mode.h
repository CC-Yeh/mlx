// Copyright © 2026 Apple Inc.
#pragma once

#include <string>
#include <string_view>

namespace mlx::core {

enum class QuantizationMode { Affine, Mxfp4, Mxfp8, Nvfp4 };

std::string quantization_mode_to_string(QuantizationMode mode);
QuantizationMode string_to_quantization_mode(
    const std::string& mode,
    std::string_view error_tag = "");

} // namespace mlx::core
