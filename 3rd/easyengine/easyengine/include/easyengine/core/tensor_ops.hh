#pragma once
#include <vector>
#include "easyengine/core/core.hh"
#include "easyengine/core/stream.hh"

namespace easyengine {

namespace functions {
void concat_tensor(
    cudaStream_t stream,
    const core::Tensor& A,
    const core::Tensor& B,
    core::Tensor& out,
    int dim = -1);
void index_select(
    cudaStream_t stream,
    const core::Tensor& input,
    int dim,
    const core::Tensor& index, // the 1-D tensor containing the indices to index,
    core::Tensor& out);
} // namespace functions
} // namespace easyengine