#pragma once

// @generated by torchgen/gen.py from Operator.h

#include <string_view>
#include <tuple>
#include <vector>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {
namespace _ops {


struct TORCH_API _nested_compute_contiguous_strides_offsets {
  using schema = ::std::tuple<at::Tensor,at::Tensor> (const at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  static constexpr const char* name = "aten::_nested_compute_contiguous_strides_offsets";
  static constexpr const char* overload_name = "";
  static constexpr const char* schema_str = "_nested_compute_contiguous_strides_offsets(Tensor nested_size) -> (Tensor, Tensor)";
  static ::std::tuple<at::Tensor,at::Tensor> call(const at::Tensor & nested_size);
  static ::std::tuple<at::Tensor,at::Tensor> redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & nested_size);
};

}} // namespace at::_ops
