#include "nn/attention/attention_kernel.h"

#include "nn/functions/functions.h"
#include <bmengine/core/core.h>
#include <bmengine/functions/utils.cuh>
#include <bmengine/functions/reduce.cuh>
#include <bmengine/functions/softmax.h>
#include <bmengine/logger/std_log_op.hpp>
#include "utils/env.h"
#include <iostream>
#include <assert.h>
#include <cuda.h>
#include <vector_types.h>

namespace nn {

using bmengine::core::DataType;
using bmengine::core::Tensor;

// gridDim (num_heads, len_q, 1),  blockDim (1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(fused_scale_mask_bias)(
    int len_buf,
    int len_q,
    T scale,
    int score_stride,
    int mask_stride,
    const int8_t *__restrict__ mask,     // (len_q, len_buf)
    const T *__restrict__ position_bias, // (num_heads, len_q, len_buf)
    T *__restrict__ x                    // (num_heads, len_q, len_buf)
) {
    int batch_id = blockIdx.z;
    int offset = batch_id * score_stride + (blockIdx.x * len_q + blockIdx.y) * len_buf;

    for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
        x[offset + i] = mask[batch_id * mask_stride + blockIdx.y * len_buf + i] > 0
                        ? x[offset + i] * scale + position_bias[offset + i]
                        : -functions::Inf<T>();
    }
}

// gridDim (num_heads, len_q, 1),  blockDim (1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(fused_scale_mask)(
    size_t len_buf,
    size_t len_q,
    T scale,
    size_t score_stride,
    size_t mask_stride,
    const int8_t *__restrict__ mask, // (len_q, len_buf)
    T *__restrict__ x                // (num_heads, len_q, len_buf)
) {
    size_t batch_id = blockIdx.z;
    x += batch_id * score_stride + (blockIdx.x * len_q + blockIdx.y) * len_buf;
    mask += batch_id * mask_stride + blockIdx.y * len_buf;

    for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
        x[i] = mask[i] > 0 ? x[i] * scale : -functions::Inf<T>();
    }
}

// gridDim (num_heads, len_q, batch),  blockDim (1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(fused_scale_mask_bias_softmax)(
    int len_buf,
    int len_q,
    T scale,
    int score_stride,
    int mask_stride,
    const int8_t *__restrict__ mask,     // (batch, len_q, len_buf)
    const T *__restrict__ position_bias, // (batch, num_heads, len_q, len_buf)
    T *__restrict__ x                    // (batch, num_heads, len_q, len_buf)
) {
    int batch_id = blockIdx.z;
    int offset = batch_id * score_stride + (blockIdx.x * len_q + blockIdx.y) * len_buf;
    int offset_m = batch_id * mask_stride + blockIdx.y * len_buf;
    functions::SharedMemory<float> shared;
    float *smem = shared.getPointer();

    for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
        smem[i] = mask[offset_m + i] > 0 ? x[offset + i] * scale + position_bias[offset + i]
                                         : -functions::Inf<T>();
    }
    float local_max = -1e20;
    for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
        local_max = fmaxf(local_max, smem[i]);
    }
    local_max = functions::blockReduceMax<float>(local_max);

    float local_sum = 1e-20;
    for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
        float v = expf(float(smem[i]) - local_max);
        smem[i] = v;
        local_sum += v;
    }
    local_sum = functions::blockReduceSum<float>(local_sum);
    for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
        x[offset + i] = float(smem[i]) / local_sum;
    }
}

// gridDim (num_heads, len_q, batch),  blockDim (len_buf~1024, 1, 1)
template<typename T>
static __global__ void BM_KERNEL(fused_scale_mask_softmax)(
    int len_buf,
    size_t len_q,
    T scale,
    size_t score_stride,
    size_t mask_stride,
    const int8_t *__restrict__ mask, // (batch, len_q, len_buf)
    T *__restrict__ x                // (batch, num_heads, len_q, len_buf)
) {
    const float NEG_INFINITY = -functions::Inf<float>();
    int batch_id = blockIdx.z;
    x += batch_id * score_stride + (blockIdx.x * len_q + blockIdx.y) * len_buf;
    mask += batch_id * mask_stride + blockIdx.y * len_buf;

    functions::SharedMemory<float> shared;
    float *smem = shared.getPointer(); // len_buf
    for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
        smem[i] = mask[i] > 0 ? float(x[i] * scale) : NEG_INFINITY;
    }
    float local_max = -1e20;
    for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
        local_max = fmaxf(local_max, smem[i]);
    }
    local_max = functions::blockReduceMax<float>(local_max);

    float local_sum = 1e-20;
    for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
        if (smem[i] == NEG_INFINITY) {
            smem[i] = 0;
        } else {
            float v = expf(float(smem[i]) - local_max);
            smem[i] = v;
            local_sum += v;
        }
    }
    local_sum = functions::blockReduceSum<float>(local_sum);
    for (int i = threadIdx.x; i < len_buf; i += blockDim.x) {
        x[i] = float(smem[i]) / local_sum;
    }
}

void attn_softmax(
    const core::Context& ctx,
    float scale,
    const core::Tensor& attn_score, // (batch?, num_heads, len_q, len_buf)
    const core::Tensor& mask,       // (batch?, len_q, len_buf)
    const core::Tensor&
    position_bias // if relative (batch?, num_head, len_q, len_buf) else if core::Tensor()
) {
    auto dtype = attn_score.dtype();
    int batch = (attn_score.ndim() <= 3) ? 1 : attn_score.size(0);
    int score_stride = (attn_score.ndim() <= 3) ? 0 : attn_score.stride(0);
    int mask_stride = (mask.ndim() <= 2) ? 0 : mask.stride(0);
    int num_heads = attn_score.size(-3);
    int len_q = attn_score.size(-2);
    int len_buf = attn_score.size(-1);
    if (position_bias.numel() > 0) {
        BM_ASSERT_EQ(attn_score.numel(), position_bias.numel(), "shape mismatch");
    }

    dim3 gridDim(num_heads, len_q, batch);
    dim3 blockDim(min(1024, round_up(len_buf, 32)), 1, 1);
    auto stream = ctx.current_stream()->ptr;

    BM_DTYPE_DISPATCH_FLOAT(dtype, {
        size_t dynamic_size = len_buf * sizeof(float);
        if (dynamic_size < 48 * 1000 && position_bias.numel() > 0) {
            BM_KERNEL(fused_scale_mask_bias_softmax)<scalar_t>
            <<<gridDim, blockDim, dynamic_size, stream>>>(
                len_buf,
                    len_q,
                    scale,
                    score_stride,
                    mask_stride,
                    mask.data<int8_t>(),
                    position_bias.data<scalar_t>(),
                    attn_score.data<scalar_t>());
        } else if (dynamic_size < ctx.get_max_shared_memory() && position_bias.numel() == 0) {
            if (dynamic_size > 48 * 1000) {
                BM_CUDART_ASSERT(cudaFuncSetAttribute(
                    BM_KERNEL(fused_scale_mask_softmax)<scalar_t> ,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    dynamic_size));
            }
            BM_KERNEL(fused_scale_mask_softmax)<scalar_t>
            <<<gridDim, blockDim, dynamic_size, stream>>>(
                len_buf,
                    len_q,
                    scale,
                    score_stride,
                    mask_stride,
                    mask.data<int8_t>(),
                    attn_score.data<scalar_t>());
        } else {
            // BM_ASSERT(false, "dynamic_size too big");
            if (position_bias.numel() > 0) {
                BM_KERNEL(fused_scale_mask_bias)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
                    len_buf,
                        len_q,
                        scale,
                        score_stride,
                        mask_stride,
                        mask.data<int8_t>(),
                        position_bias.data<scalar_t>(),
                        attn_score.data<scalar_t>());
            } else {
                BM_KERNEL(fused_scale_mask)<scalar_t><<<gridDim, blockDim, 0, stream>>>(
                    len_buf,
                        len_q,
                        scale,
                        score_stride,
                        mask_stride,
                        mask.data<int8_t>(),
                        attn_score.data<scalar_t>());
                BM_CUDART_ASSERT(cudaGetLastError());
            }
            softmax(ctx, attn_score, attn_score, 1.0);
        }
    });
    BM_CUDART_ASSERT(cudaGetLastError());
}

} // namespace nn
