/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Separate from unfusedAttentionKernel to accelerate compiling.

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttentionUtils.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

#define WARP_SIZE 32
#define HALF_WARP_SIZE 16
#define WARPS_PER_SM 32
#define MIN_SEQUENCES_PER_WARP 4
#define WARPS_PER_BLOCK 32

////////////////////////////////////////////////////////////////////////////////////////////////////

// One warp of threads handle one head in terms of rotary embedding.
// Balance the work across threads in one warp for different head size.
// The minimum head size should be 32.
// Assume head size <= 256.
template <typename T, int Dh_MAX>
struct Rotary_vec_t
{
    using Type = T;
    using BaseType = T;
    // Quantized output type only supports fp8 currently.
    using QuantizedType = __nv_fp8_e4m3;
    static constexpr int size = 1;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Rotary_vec_t<float, 32>
{
    using Type = float;
    using BaseType = float;
    using QuantizedType = __nv_fp8_e4m3;
    static constexpr int size = 1;
};

template <>
struct Rotary_vec_t<float, 64>
{
    using Type = float2;
    using BaseType = float;
    using QuantizedType = mmha::fp8_2_t;
    static constexpr int size = 2;
};

template <>
struct Rotary_vec_t<float, 128>
{
    using Type = float4;
    using BaseType = float;
    using QuantizedType = mmha::fp8_4_t;
    static constexpr int size = 4;
};

template <>
struct Rotary_vec_t<float, 256>
{
    using Type = mmha::Float8_;
    using BaseType = float;
    using QuantizedType = mmha::fp8_8_t;
    static constexpr int size = 8;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Rotary_vec_t<half, 32>
{
    using Type = uint16_t;
    using BaseType = uint16_t;
    using QuantizedType = __nv_fp8_e4m3;
    static constexpr int size = 2;
};

template <>
struct Rotary_vec_t<half, 64>
{
    using Type = uint32_t;
    using BaseType = uint16_t;
    using QuantizedType = mmha::fp8_2_t;
    static constexpr int size = 2;
};

template <>
struct Rotary_vec_t<half, 128>
{
    using Type = uint2;
    using BaseType = uint16_t;
    using QuantizedType = mmha::fp8_4_t;
    static constexpr int size = 4;
};

template <>
struct Rotary_vec_t<half, 256>
{
    using Type = uint4;
    using BaseType = uint16_t;
    using QuantizedType = mmha::fp8_8_t;
    static constexpr int size = 8;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_BF16

template <>
struct Rotary_vec_t<__nv_bfloat16, 32>
{
    using Type = __nv_bfloat16;
    using BaseType = __nv_bfloat16;
    using QuantizedType = __nv_fp8_e4m3;
    static constexpr int size = 1;
};

template <>
struct Rotary_vec_t<__nv_bfloat16, 64>
{
    using Type = __nv_bfloat162;
    using BaseType = __nv_bfloat16;
    using QuantizedType = mmha::fp8_2_t;
    static constexpr int size = 2;
};

template <>
struct Rotary_vec_t<__nv_bfloat16, 128>
{
    using Type = mmha::bf16_4_t;
    using BaseType = __nv_bfloat16;
    using QuantizedType = mmha::fp8_4_t;
    static constexpr int size = 4;
};

template <>
struct Rotary_vec_t<__nv_bfloat16, 256>
{
    using Type = mmha::bf16_8_t;
    using BaseType = __nv_bfloat16;
    using QuantizedType = mmha::fp8_8_t;
    static constexpr int size = 8;
};

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

// Multiple calls of reinterpret_cast.
template <typename type_in, typename type_out>
inline __device__ type_out* reinterpret_ptr(void* ptr, size_t offset)
{
    return reinterpret_cast<type_out*>(reinterpret_cast<type_in*>(ptr) + offset);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Make sure each thread at least processes two elements (gptj rotary embedding).

template <typename T, RotaryPositionEmbeddingType ROTARY_TYPE>
struct Rotary_base_t
{
    using RotaryBaseType = T;
};

template <>
struct Rotary_base_t<uint16_t, RotaryPositionEmbeddingType::GPTJ>
{
    using RotaryBaseType = uint32_t;
};

#ifdef ENABLE_BF16
template <>
struct Rotary_base_t<__nv_bfloat16, RotaryPositionEmbeddingType::GPTJ>
{
    using RotaryBaseType = __nv_bfloat162;
};
#endif

template <>
struct Rotary_base_t<float, RotaryPositionEmbeddingType::GPTJ>
{
    using RotaryBaseType = float2;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename VecType, typename T, int VEC_SIZE, bool RECOMPUTE>
inline __device__ void apply_rotary_embedding_gptneox(VecType& q, VecType& q_pair, VecType& k, VecType& k_pair,
    bool first_half, float2 (&rotary_coef_cache)[VEC_SIZE], float const* rotary_inv_freq_buffer,
    int const rotary_dim_idx, int const half_rotary_dim, int const rotary_position, int const vision_start = -1,
    int const vision_length = -1)
{
    // Each thread holds NUM_ELTS elements.
    // Currently we apply the rotary embedding in float data type for accuracy reasons.
    using RotaryBaseType = typename Rotary_base_t<T, RotaryPositionEmbeddingType::GPT_NEOX>::RotaryBaseType;
#pragma unroll
    for (int elt_id = 0; elt_id < VEC_SIZE; elt_id++)
    {
        RotaryBaseType& q_ = reinterpret_cast<RotaryBaseType*>(&q)[elt_id];
        RotaryBaseType q_pair_ = reinterpret_cast<RotaryBaseType*>(&q_pair)[elt_id];
        RotaryBaseType& k_ = reinterpret_cast<RotaryBaseType*>(&k)[elt_id];
        RotaryBaseType k_pair_ = reinterpret_cast<RotaryBaseType*>(&k_pair)[elt_id];

        bool const valid_rotary_pos = rotary_dim_idx + elt_id < half_rotary_dim;

        if (RECOMPUTE)
        {
            int real_rotary_position = rotary_position;
            if (vision_start != -1 && vision_length != -1)
            {
                int t_step_int = rotary_position;
                if (t_step_int <= vision_start)
                {
                    real_rotary_position = t_step_int;
                }
                else if (t_step_int > vision_start && t_step_int <= (vision_length + vision_start))
                {
                    real_rotary_position = vision_start + 1;
                }
                else
                {
                    real_rotary_position = t_step_int - (vision_length - 1);
                }
            }
            float const rotary_inv_freq = float(real_rotary_position)
                * rotary_inv_freq_buffer[min(rotary_dim_idx + elt_id, half_rotary_dim - 1)];
            rotary_coef_cache[elt_id] = make_float2(cosf(rotary_inv_freq), sinf(rotary_inv_freq));
        }

        // Mask non-rotary dim.
        float2 rotary_coef = valid_rotary_pos ? rotary_coef_cache[elt_id] : make_float2(1.0f, 0.0f);
        // Pre-process different half of rotary dimension.
        rotary_coef.y = first_half ? -rotary_coef.y : rotary_coef.y;

        mmha::apply_rotary_embedding_gptneox(q_, q_pair_, k_, k_pair_, rotary_coef);
    }
}

template <typename VecType, typename T, int VEC_SIZE, bool RECOMPUTE>
inline __device__ void apply_rotary_embedding_gptj(VecType& q, VecType& k, float2 (&rotary_coef_cache)[VEC_SIZE],
    float const* rotary_inv_freq_buffer, int const rotary_dim_idx, int const half_rotary_dim, int const rotary_position)
{
    // Each thread holds NUM_ELTS elements.
    // Currently we apply the rotary embedding in float data type for accuracy reasons.
    using RotaryBaseType = typename Rotary_base_t<T, RotaryPositionEmbeddingType::GPTJ>::RotaryBaseType;
#pragma unroll
    for (int elt_id = 0; elt_id < VEC_SIZE; elt_id++)
    {
        RotaryBaseType q_ = reinterpret_cast<RotaryBaseType*>(&q)[elt_id];
        RotaryBaseType k_ = reinterpret_cast<RotaryBaseType*>(&k)[elt_id];

        bool const valid_rotary_pos = rotary_dim_idx + elt_id < half_rotary_dim;

        if (RECOMPUTE)
        {
            float const rotary_inv_freq
                = float(rotary_position) * rotary_inv_freq_buffer[min(rotary_dim_idx + elt_id, half_rotary_dim - 1)];
            rotary_coef_cache[elt_id] = make_float2(cosf(rotary_inv_freq), sinf(rotary_inv_freq));
        }

        mmha::apply_rotary_embedding_gptj(q_, k_, rotary_coef_cache[elt_id]);

        if (valid_rotary_pos)
        {
            reinterpret_cast<RotaryBaseType*>(&q)[elt_id] = q_;
            reinterpret_cast<RotaryBaseType*>(&k)[elt_id] = k_;
        }
    }
}

template <typename T, typename TCache, int Dh_MAX, bool ADD_BIAS, bool STORE_QKV, typename KVCacheBuffer,
    RotaryPositionEmbeddingType ROTARY_TYPE, bool DYNAMIC_ROTARY_SCALING, bool FP8_OUTPUT>
__global__ void applyBiasRopeUpdateKVCache(QKVPreprocessingParams<T, KVCacheBuffer> params)
{
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head]
    // Extract the Q input when using paged KV FMHA.
    // For q and k, also apply the rotary embedding.

    // NOTE:
    // In the case of in-place modifications:
    //      head_num == kv_head_num
    //      QKV src shape (batch_size, seq_len, 3, head_num, size_per_head)
    //                      ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                              m                        n
    //      head_num != kv_head_num
    //      QKV src shape: (batch_size, seq_len, head_num * size_per_head + 2 * kv_head_num * size_per_head)
    //                      ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                                 m                               n
    // Additionally, if there is no padding:
    //      QKV src shape (num_tokens, 3, head_num, size_per_head)
    //
    //  In these above cases, output shape stays the same
    //
    //   Q dst shape: (batch_size, head_num, seq_len, size_per_head)
    //   KV dst shape: (batch_size, kv_head_num, seq_len, size_per_head)
    // See QKVPreprocessingParams for further details

    // There are two kinds of output:
    //  1. Contiguous QKV output.
    //  2. Contiguous Q output + Paged KV output (needed by Paged KV FMHA kernels).

    // VEC_SIZE is power of 2.
    constexpr int VEC_SIZE = Rotary_vec_t<T, Dh_MAX>::size;
    using VecType = typename Rotary_vec_t<T, Dh_MAX>::Type;
    // The base type will share the rotary coefficient.
    using BaseType = typename Rotary_vec_t<T, Dh_MAX>::BaseType;
    // Quantized output only supports fp8 currently.
    using QuantizedEltType = __nv_fp8_e4m3;
    using QuantizedVecType = typename Rotary_vec_t<T, Dh_MAX>::QuantizedType;
    // GPTJ rotary embedding: two elements share the same rotary coefficient.
    constexpr int ROTARY_COEF_VEC_SIZE = ROTARY_TYPE == RotaryPositionEmbeddingType::GPTJ ? VEC_SIZE / 2 : VEC_SIZE;

    constexpr bool ENABLE_8BITS_CACHE = sizeof(TCache) == 1;
    int const sizePerHeadDivX = params.size_per_head / VEC_SIZE;
    using TDst = TCache;

    // Variable sequence length.
    bool const variable_sequence_length = params.cu_seq_lens != nullptr;

    int const head_idx = blockIdx.y;
    // Block size is always 32 in the x dimension (handles one head size).
    int const tidx = threadIdx.x;
    int const head_dim_idx = tidx * VEC_SIZE;
    bool const first_half = head_dim_idx < params.half_rotary_dim;
    int const rotated_head_dim_offset = first_half ? params.half_rotary_dim : -params.half_rotary_dim;
    int const gptneox_rotary_dim_idx = first_half ? head_dim_idx : (head_dim_idx - params.half_rotary_dim);
    int const gptj_rotary_dim_idx = head_dim_idx / 2;

    int const hidden_idx = head_idx * params.size_per_head + head_dim_idx;
    int const kv_head_idx = head_idx / params.qheads_per_kv_head;
    int const hidden_idx_kv = kv_head_idx * params.size_per_head + head_dim_idx;
    int const hidden_size = params.hidden_size;
    int const src_k_offset = params.q_hidden_size;
    int const src_v_offset = src_k_offset + params.kv_head_num * params.size_per_head;

    // Reuse the rotary coefficients for the same rotary position.
    float2 rotary_coef_cache[ROTARY_COEF_VEC_SIZE];

    int local_token_idx = blockIdx.x * blockDim.y + threadIdx.y;
    {
        int cached_rotary_position = -1;
        for (int batch_idx = blockIdx.z; batch_idx < params.batch_size; batch_idx += gridDim.z)
        {
            // The index of the token in the batch.
            int const global_token_idx = local_token_idx
                + ((variable_sequence_length && params.remove_padding) ? params.cu_seq_lens[batch_idx]
                                                                       : batch_idx * params.max_input_seq_len);
            int const cache_seq_len = params.cache_seq_lens[batch_idx];
            int const actual_seq_len = variable_sequence_length ? params.seq_lens[batch_idx] : params.max_input_seq_len;
            // Chunked attention: takes past_kv_sequence_length into consideration.
            int const token_idx_in_seq = (cache_seq_len - actual_seq_len) + local_token_idx;
            bool const valid_token = token_idx_in_seq < cache_seq_len;

            // NOTE: only spec decoding needs the position offsets.
            // In the generation phase, we assume all sequences should have the same input length.
            int const rotary_position = params.spec_decoding_position_offsets != nullptr
                ? (params.spec_decoding_position_offsets[local_token_idx + batch_idx * params.max_input_seq_len]
                    + cache_seq_len - actual_seq_len)
                : token_idx_in_seq;

            if (!valid_token)
            {
                continue;
            }

            // Is the token and head dim maksed.
            bool const valid_head_dim_idx = head_dim_idx < params.size_per_head;

            // head_num == kv_head_num:
            //   src QKV: [batch, time, 3, head_num, size_per_head]
            // head_num != kv_head_num:
            //   src QKV: [batch, time, head_num * size_per_head + 2 * kv_head_num * size_per_head]
            auto const src_q_idx = static_cast<size_t>(global_token_idx) * hidden_size + hidden_idx;
            auto const src_k_idx = static_cast<size_t>(global_token_idx) * hidden_size + src_k_offset + hidden_idx_kv;
            auto const src_v_idx = static_cast<size_t>(global_token_idx) * hidden_size + src_v_offset + hidden_idx_kv;

            VecType q, k, v, q_pair, k_pair;
            // key without position embedding
            VecType k_wo_pos;

            // load q,k,v and add bias
            if (valid_head_dim_idx)
            {
                q = *reinterpret_cast<VecType const*>(&params.qkv_input[src_q_idx]);
                k = *reinterpret_cast<VecType const*>(&params.qkv_input[src_k_idx]);
                v = *reinterpret_cast<VecType const*>(&params.qkv_input[src_v_idx]);
                q_pair = *reinterpret_cast<VecType const*>(&params.qkv_input[src_q_idx + rotated_head_dim_offset]);
                k_pair = *reinterpret_cast<VecType const*>(&params.qkv_input[src_k_idx + rotated_head_dim_offset]);

                if constexpr (ADD_BIAS)
                {
                    auto const q_bias = *reinterpret_cast<VecType const*>(&params.qkv_bias[hidden_idx]);
                    auto const k_bias
                        = *reinterpret_cast<VecType const*>(&params.qkv_bias[hidden_idx_kv + src_k_offset]);
                    auto const v_bias
                        = *reinterpret_cast<VecType const*>(&params.qkv_bias[hidden_idx_kv + src_v_offset]);
                    auto const q_pair_bias
                        = *reinterpret_cast<VecType const*>(&params.qkv_bias[hidden_idx + rotated_head_dim_offset]);
                    auto const k_pair_bias = *reinterpret_cast<VecType const*>(
                        &params.qkv_bias[hidden_idx_kv + src_k_offset + rotated_head_dim_offset]);

                    q = mmha::add(q, q_bias);
                    k = mmha::add(k, k_bias);
                    v = mmha::add(v, v_bias);
                    q_pair = mmha::add(q_pair, q_pair_bias);
                    k_pair = mmha::add(k_pair, k_pair_bias);
                }
                k_wo_pos = k;
            }

            switch (ROTARY_TYPE)
            {
            // Rotate every two elements (need at two elements per thead).
            // e.g.  0  1  2  3  4  5  6  7 (head size 8)
            //      -1  0 -3  2 -5  4 -7  6
            case RotaryPositionEmbeddingType::GPTJ:
            {
                if (DYNAMIC_ROTARY_SCALING || rotary_position != cached_rotary_position)
                {
                    apply_rotary_embedding_gptj<VecType, BaseType, ROTARY_COEF_VEC_SIZE, true>(q, k, rotary_coef_cache,
                        params.rotary_embedding_inv_freq + batch_idx * params.half_rotary_dim, gptj_rotary_dim_idx,
                        params.half_rotary_dim, rotary_position);
                    cached_rotary_position = rotary_position;
                }
                else
                {
                    apply_rotary_embedding_gptj<VecType, BaseType, ROTARY_COEF_VEC_SIZE, false>(q, k, rotary_coef_cache,
                        params.rotary_embedding_inv_freq + batch_idx * params.half_rotary_dim, gptj_rotary_dim_idx,
                        params.half_rotary_dim, rotary_position);
                }
                break;
            }
            // Rotate by half rotary embedding.
            // e.g.  0  1  2  3  4  5  6  7 (head size 8)
            //      -4 -5 -6 -7  0  1  2  3
            case RotaryPositionEmbeddingType::GPT_NEOX:
            {
                if (DYNAMIC_ROTARY_SCALING || rotary_position != cached_rotary_position)
                {
                    apply_rotary_embedding_gptneox<VecType, BaseType, ROTARY_COEF_VEC_SIZE, true>(q, q_pair, k, k_pair,
                        first_half, rotary_coef_cache,
                        params.rotary_embedding_inv_freq + batch_idx * params.half_rotary_dim, gptneox_rotary_dim_idx,
                        params.half_rotary_dim, rotary_position, params.rotary_vision_start,
                        params.rotary_vision_length);
                    cached_rotary_position = rotary_position;
                }
                else
                {
                    apply_rotary_embedding_gptneox<VecType, BaseType, ROTARY_COEF_VEC_SIZE, false>(q, q_pair, k, k_pair,
                        first_half, rotary_coef_cache,
                        params.rotary_embedding_inv_freq + batch_idx * params.half_rotary_dim, gptneox_rotary_dim_idx,
                        params.half_rotary_dim, rotary_position, params.rotary_vision_start,
                        params.rotary_vision_length);
                }
                break;
            }
            }

            auto const channelIdx{tidx};
            auto const tokenIdxLowerBound
                = max(cache_seq_len - params.cyclic_kv_cache_len + params.sink_token_len, params.sink_token_len);
            bool const valid_kv_cache_pos
                = params.kv_cache_buffer.data != nullptr // In KV-cache-less mode. No need to store KV values
                && (token_idx_in_seq >= tokenIdxLowerBound || token_idx_in_seq < params.sink_token_len);
            auto const token_kv_idx = params.kv_cache_buffer.getKVTokenIdx(token_idx_in_seq);

            // Make sure pairs of q or v vecs have been read before write.
            // One block will handle single head.
            __syncthreads();

            if (valid_head_dim_idx)
            {
                auto kDst = reinterpret_cast<TDst*>(params.kv_cache_buffer.getKBlockPtr(batch_idx, token_kv_idx));
                auto vDst = reinterpret_cast<TDst*>(params.kv_cache_buffer.getVBlockPtr(batch_idx, token_kv_idx));
                int inBlockIdx
                    = params.kv_cache_buffer.getKVLocalIdx(token_kv_idx, kv_head_idx, sizePerHeadDivX, channelIdx);
                VecType k_to_cache = params.position_shift_enabled ? k_wo_pos : k;

                auto const dst_q_idx = static_cast<size_t>(global_token_idx) * params.q_hidden_size + hidden_idx;
                VecType* q_ptr = STORE_QKV ? reinterpret_ptr<T, VecType>(params.qkv_input, src_q_idx)
                                           : reinterpret_ptr<T, VecType>(params.q_output, dst_q_idx);

                // Cast float scale to dst data type.
                using TScale = typename mmha::kv_cache_scale_type_t<T, TCache>::Type;
                [[maybe_unused]] TScale scaleOrigQuant;
                if constexpr (FP8_OUTPUT || ENABLE_8BITS_CACHE)
                {
                    mmha::convert_from_float(
                        &scaleOrigQuant, params.kvScaleOrigQuant ? params.kvScaleOrigQuant[0] : 1.0f);
                }

                if constexpr (FP8_OUTPUT)
                {
                    // Quant the vec to fp8 vec with the scale.
                    QuantizedEltType* quantized_q_ptr = STORE_QKV
                        ? reinterpret_cast<QuantizedEltType*>(params.quantized_qkv_output) + src_q_idx
                        : reinterpret_cast<QuantizedEltType*>(params.q_output) + dst_q_idx;
                    mmha::store_8bits_vec(quantized_q_ptr, q, 0, scaleOrigQuant);
                }
                else
                {
                    *q_ptr = q;
                }
                if ((params.head_num == params.kv_head_num) || (head_idx == (kv_head_idx * params.qheads_per_kv_head)))
                {
                    if constexpr (STORE_QKV)
                    {
                        if constexpr (FP8_OUTPUT)
                        {
                            // Quant the vec to fp8 vec with the scale.
                            mmha::store_8bits_vec(reinterpret_cast<QuantizedEltType*>(params.quantized_qkv_output), k,
                                src_k_idx, scaleOrigQuant);
                            mmha::store_8bits_vec(reinterpret_cast<QuantizedEltType*>(params.quantized_qkv_output), v,
                                src_v_idx, scaleOrigQuant);
                        }
                        else
                        {
                            *reinterpret_cast<VecType*>(&params.qkv_input[src_k_idx]) = k;
                            if constexpr (ADD_BIAS)
                            {
                                *reinterpret_cast<VecType*>(&params.qkv_input[src_v_idx]) = v;
                            }
                        }
                    }

                    if (valid_kv_cache_pos)
                    {
                        if constexpr (ENABLE_8BITS_CACHE)
                        {
                            inBlockIdx = inBlockIdx * VEC_SIZE;
                            // Store 8bits kv cache.
                            mmha::store_8bits_vec(kDst, k_to_cache, inBlockIdx, scaleOrigQuant);
                            mmha::store_8bits_vec(vDst, v, inBlockIdx, scaleOrigQuant);
                        }
                        else
                        {
                            reinterpret_cast<VecType*>(kDst)[inBlockIdx] = k_to_cache;
                            reinterpret_cast<VecType*>(vDst)[inBlockIdx] = v;
                        }
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Bandwidth-bound kernel by reading cos/sin coefficients from global memory (pre-computed and saved as weights).

template <typename T>
struct VecType
{
    using Type = T;
    using QuantizedType = mmha::fp8_8_t;
    using GPTNeoXEltType = T;
    using GPTJEltType = T;
};

template <>
struct VecType<float>
{
    using Type = float4;
    using QuantizedType = mmha::fp8_4_t;
    using GPTNeoXEltType = float;
    using GPTJEltType = float2;
};

template <>
struct VecType<half>
{
    using Type = uint4;
    using QuantizedType = mmha::fp8_8_t;
    using GPTNeoXEltType = uint16_t;
    using GPTJEltType = uint32_t;
};

template <>
struct VecType<__nv_bfloat16>
{
    using Type = mmha::bf16_8_t;
    using QuantizedType = mmha::fp8_8_t;
    using GPTNeoXEltType = __nv_bfloat16;
    using GPTJEltType = __nv_bfloat162;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename TCache, int BLOCK_SIZE, int Dh, bool ADD_BIAS, bool STORE_QKV, bool FP8_OUTPUT,
    typename KVCacheBuffer, RotaryPositionEmbeddingType ROTARY_TYPE>
__global__ void applyBiasRopeUpdateKVCacheV2(QKVPreprocessingParams<T, KVCacheBuffer> params)
{
    // This kernel add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head]
    // Extract the Q input when using paged KV FMHA.
    // For q and k, also apply the rotary embedding.

    // NOTE:
    // head_num == kv_head_num
    //   QKV src shape (batch_size, seq_len, 3, head_num, size_per_head)
    //                  ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                           m                        n
    //   QKV dst shape (3, batch_size, head_num, seq_len, size_per_head)
    // head_num != kv_head_num
    //   QKV src shape: (batch_size, seq_len, head_num * size_per_head + 2 * kv_head_num * size_per_head)
    //                   ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                             m                               n
    // Additionally, if there is no padding:
    //   QKV src shape (num_tokens, 3, head_num, size_per_head)
    //
    //   Q dst shape: (batch_size, head_num, seq_len, size_per_head)
    //   KV dst shape: (batch_size, kv_head_num, seq_len, size_per_head)
    // See QKVPreprocessingParams for further details

    // There are two kinds of output:
    //  1. Contiguous QKV output.
    //  2. Contiguous Q output + Paged KV output (needed by Paged KV FMHA kernels).

    // Constants.
    using VecT = typename VecType<T>::Type;
    // Quantized output only supports fp8 currently.
    using QuantizedEltType = __nv_fp8_e4m3;
    using QuantizedVecType = typename VecType<T>::QuantizedType;
    using GPTNeoXEltT = typename VecType<T>::GPTNeoXEltType;
    using GPTJEltT = typename VecType<T>::GPTJEltType;
    constexpr auto HEAD_SIZE = Dh;
    constexpr auto BYTES_PER_ELT = sizeof(T);
    constexpr auto BYTES_PER_LOAD = 16;
    constexpr auto ELTS_PER_VEC = BYTES_PER_LOAD / BYTES_PER_ELT;
    static_assert((HEAD_SIZE * BYTES_PER_ELT) % BYTES_PER_LOAD == 0, "Head size needs to be multiple of 16 bytes.");
    constexpr auto VECS_PER_HEAD = HEAD_SIZE * BYTES_PER_ELT / BYTES_PER_LOAD;
    static_assert(BLOCK_SIZE % VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");
    constexpr auto TOKENS_PER_BLOCK = BLOCK_SIZE / VECS_PER_HEAD;
    // int8 / fp8 kv cache.
    constexpr bool ENABLE_8BITS_CACHE = sizeof(TCache) == 1;

    // Block/Head idx.
    int const batch_idx = blockIdx.y;
    int const head_idx = blockIdx.z;

    // Variable sequence length.
    bool const variable_sequence_length = params.cu_seq_lens != nullptr;
    int const head_dim_vec_idx = (threadIdx.x % VECS_PER_HEAD);
    int const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;
    bool const first_half = head_dim_idx < params.half_rotary_dim;
    [[maybe_unused]] int const gptneox_rotary_dim_idx
        = first_half ? head_dim_idx : (head_dim_idx - params.half_rotary_dim);
    [[maybe_unused]] int const gptj_rotary_dim_idx = head_dim_idx / 2;
    // Assume that either all vector elements are valid rotary idx or not.
    [[maybe_unused]] int const valid_rotary_dim_idx = head_dim_idx < params.rotary_embedding_dim;
    float2 const masked_rotary_cos_sin = make_float2(1.0f, 0.0f);
    int const hidden_idx = head_idx * params.size_per_head + head_dim_idx;
    int const kv_head_idx = head_idx / params.qheads_per_kv_head;
    int const hidden_idx_kv = kv_head_idx * params.size_per_head + head_dim_idx;
    int const src_k_offset = params.q_hidden_size;
    int const src_v_offset = src_k_offset + params.kv_hidden_size;

    int const rotated_head_dim_offset = first_half ? params.half_rotary_dim : -params.half_rotary_dim;
    // Make sure there are multiple of tokens_per_block otherwise syncthreads will lead to deadlocks.
    int const seq_len_loop_end
        = int((params.max_input_seq_len + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK;

    // Mainloop.
    for (int local_token_idx = (threadIdx.x / VECS_PER_HEAD) + blockIdx.x * TOKENS_PER_BLOCK;
         local_token_idx < seq_len_loop_end; local_token_idx += TOKENS_PER_BLOCK * gridDim.x)
    {
        // The index of the token in the batch.
        int const global_token_offset = (variable_sequence_length && params.remove_padding)
            ? params.cu_seq_lens[batch_idx]
            : batch_idx * params.max_input_seq_len;
        int const cache_seq_len = params.cache_seq_lens[batch_idx];
        int const actual_seq_len = variable_sequence_length ? params.seq_lens[batch_idx] : params.max_input_seq_len;
        // Chunked attention: takes past_kv_sequence_length into consideration.
        int const past_seq_len = (cache_seq_len - actual_seq_len);
        int token_idx_in_kv_cache = past_seq_len + local_token_idx;
        // The same as local_token_idx < actual_seq_len.
        bool const valid_token = token_idx_in_kv_cache < cache_seq_len;
        // Limit the token_idx to cache seq length (we need all threads in this block to be involved).
        token_idx_in_kv_cache = std::min(token_idx_in_kv_cache, cache_seq_len - 1);
        local_token_idx = std::min(local_token_idx, actual_seq_len - 1);
        int const global_token_idx = local_token_idx + global_token_offset;

        // NOTE: only spec decoding needs the position offsets.
        // In the generation phase, we assume all sequences should have the same input length.
        int const rotary_position = params.spec_decoding_position_offsets != nullptr
            ? (params.spec_decoding_position_offsets[local_token_idx + batch_idx * params.max_input_seq_len]
                + cache_seq_len - actual_seq_len)
            : token_idx_in_kv_cache;

        // head_num == kv_head_num:
        //   src QKV: [batch, time, 3, head_num, size_per_head]
        // head_num != kv_head_num:
        //   src QKV: [batch, time, head_num * size_per_head + 2 * kv_head_num * size_per_head]
        auto const src_q_idx = static_cast<size_t>(global_token_idx) * params.hidden_size + hidden_idx;
        auto const src_k_idx
            = static_cast<size_t>(global_token_idx) * params.hidden_size + src_k_offset + hidden_idx_kv;
        auto const src_v_idx
            = static_cast<size_t>(global_token_idx) * params.hidden_size + src_v_offset + hidden_idx_kv;

        auto q = *reinterpret_cast<VecT const*>(&params.qkv_input[src_q_idx]);
        auto k = *reinterpret_cast<VecT const*>(&params.qkv_input[src_k_idx]);
        auto v = *reinterpret_cast<VecT const*>(&params.qkv_input[src_v_idx]);
        [[maybe_unused]] auto q_pair
            = *reinterpret_cast<VecT const*>(&params.qkv_input[src_q_idx + rotated_head_dim_offset]);
        [[maybe_unused]] auto k_pair
            = *reinterpret_cast<VecT const*>(&params.qkv_input[src_k_idx + rotated_head_dim_offset]);

        // Bias should have been fused with QKV projection, but we keep the logic here for unit tests.
        if constexpr (ADD_BIAS)
        {
            auto const q_bias = *reinterpret_cast<VecT const*>(&params.qkv_bias[hidden_idx]);
            auto const k_bias = *reinterpret_cast<VecT const*>(&params.qkv_bias[hidden_idx_kv + src_k_offset]);
            auto const v_bias = *reinterpret_cast<VecT const*>(&params.qkv_bias[hidden_idx_kv + src_v_offset]);
            auto const q_pair_bias
                = *reinterpret_cast<VecT const*>(&params.qkv_bias[hidden_idx + rotated_head_dim_offset]);
            auto const k_pair_bias = *reinterpret_cast<VecT const*>(
                &params.qkv_bias[hidden_idx_kv + src_k_offset + rotated_head_dim_offset]);

            q = mmha::add(q, q_bias);
            k = mmha::add(k, k_bias);
            v = mmha::add(v, v_bias);
            q_pair = mmha::add(q_pair, q_pair_bias);
            k_pair = mmha::add(k_pair, k_pair_bias);
        }

        // Cos/sin cache.
        [[maybe_unused]] float2 const* rotary_coef_cache_buffer
            = params.rotary_coef_cache_buffer + static_cast<size_t>(rotary_position) * params.half_rotary_dim;
        if constexpr (ROTARY_TYPE == RotaryPositionEmbeddingType::GPT_NEOX)
        {
            rotary_coef_cache_buffer += gptneox_rotary_dim_idx;
#pragma unroll
            for (int elt_id = 0; elt_id < ELTS_PER_VEC; elt_id++)
            {
                GPTNeoXEltT& q_ = reinterpret_cast<GPTNeoXEltT*>(&q)[elt_id];
                GPTNeoXEltT q_pair_ = reinterpret_cast<GPTNeoXEltT*>(&q_pair)[elt_id];
                GPTNeoXEltT& k_ = reinterpret_cast<GPTNeoXEltT*>(&k)[elt_id];
                GPTNeoXEltT k_pair_ = reinterpret_cast<GPTNeoXEltT*>(&k_pair)[elt_id];

                // Load cos/sin from cache.
                float2 rotary_coef_cache
                    = valid_rotary_dim_idx ? rotary_coef_cache_buffer[elt_id] : masked_rotary_cos_sin;

                // Preprocess sin for second half rotary dim.
                rotary_coef_cache.y = first_half ? -rotary_coef_cache.y : rotary_coef_cache.y;
                mmha::apply_rotary_embedding_gptneox(q_, q_pair_, k_, k_pair_, rotary_coef_cache);
            }
        }
        else if constexpr (ROTARY_TYPE == RotaryPositionEmbeddingType::GPTJ)
        {
            rotary_coef_cache_buffer += gptj_rotary_dim_idx;
// Pack two elements into one for gptj rotary embedding.
#pragma unroll
            for (int elt_id = 0; elt_id < ELTS_PER_VEC / 2; elt_id++)
            {
                GPTJEltT& q_ = reinterpret_cast<GPTJEltT*>(&q)[elt_id];
                GPTJEltT& k_ = reinterpret_cast<GPTJEltT*>(&k)[elt_id];

                // Load cos/sin from cache.
                float2 rotary_coef_cache
                    = valid_rotary_dim_idx ? rotary_coef_cache_buffer[elt_id] : masked_rotary_cos_sin;
                mmha::apply_rotary_embedding_gptj(q_, k_, rotary_coef_cache);
            }
        }

        auto const channelIdx = head_dim_vec_idx;
        auto const tokenIdxLowerBound = max(cache_seq_len - params.cyclic_kv_cache_len, 0);
        bool const cyclic_kv_cache = cache_seq_len > params.cyclic_kv_cache_len;
        bool const useKVCache = params.kv_cache_buffer.data != nullptr;
        bool const valid_kv_cache_pos = useKVCache // In KV-cache-less mode. No need to store KV values
            && (token_idx_in_kv_cache >= tokenIdxLowerBound);
        auto const token_kv_idx
            = cyclic_kv_cache ? (token_idx_in_kv_cache % params.cyclic_kv_cache_len) : token_idx_in_kv_cache;

        auto kDst = useKVCache ? reinterpret_cast<TCache*>(params.kv_cache_buffer.getKBlockPtr(batch_idx, token_kv_idx))
                               : (TCache*) (nullptr);
        auto vDst = useKVCache ? reinterpret_cast<TCache*>(params.kv_cache_buffer.getVBlockPtr(batch_idx, token_kv_idx))
                               : (TCache*) (nullptr);
        auto inBlockIdx = useKVCache
            ? params.kv_cache_buffer.getKVLocalIdx(token_kv_idx, kv_head_idx, VECS_PER_HEAD, channelIdx)
            : int32_t(0);

        // Make sure pairs of q or v vecs have been read before write.
        __syncthreads();

        // Only update valid tokens.
        if (valid_token)
        {
            auto const dst_q_idx = static_cast<size_t>(global_token_idx) * params.q_hidden_size + hidden_idx;
            VecT* q_ptr = STORE_QKV ? reinterpret_ptr<T, VecT>(params.qkv_input, src_q_idx)
                                    : reinterpret_ptr<T, VecT>(params.q_output, dst_q_idx);

            // Cast float scale to dst data type.
            using TScale = typename mmha::kv_cache_scale_type_t<T, TCache>::Type;
            [[maybe_unused]] TScale scaleOrigQuant;
            if constexpr (FP8_OUTPUT || ENABLE_8BITS_CACHE)
            {
                mmha::convert_from_float(&scaleOrigQuant, params.kvScaleOrigQuant ? params.kvScaleOrigQuant[0] : 1.0f);
            }

            if constexpr (FP8_OUTPUT)
            {
                // Quant the vec to fp8 vec with the scale.
                QuantizedEltType* quantized_q_ptr = STORE_QKV
                    ? reinterpret_cast<QuantizedEltType*>(params.quantized_qkv_output) + src_q_idx
                    : reinterpret_cast<QuantizedEltType*>(params.q_output) + dst_q_idx;
                mmha::store_8bits_vec(quantized_q_ptr, q, 0, scaleOrigQuant);
            }
            else
            {
                *q_ptr = q;
            }
            if ((params.head_num == params.kv_head_num) || (head_idx == (kv_head_idx * params.qheads_per_kv_head)))
            {
                if constexpr (STORE_QKV)
                {
                    if constexpr (FP8_OUTPUT)
                    {
                        // Quant the vec to fp8 vec with the scale.
                        mmha::store_8bits_vec(reinterpret_cast<QuantizedEltType*>(params.quantized_qkv_output), k,
                            src_k_idx, scaleOrigQuant);
                        mmha::store_8bits_vec(reinterpret_cast<QuantizedEltType*>(params.quantized_qkv_output), v,
                            src_v_idx, scaleOrigQuant);
                    }
                    else
                    {
                        *reinterpret_cast<VecT*>(&params.qkv_input[src_k_idx]) = k;
                        if constexpr (ADD_BIAS)
                        {
                            *reinterpret_cast<VecT*>(&params.qkv_input[src_v_idx]) = v;
                        }
                    }
                }

                if (valid_kv_cache_pos)
                {
                    if constexpr (ENABLE_8BITS_CACHE)
                    {
                        inBlockIdx = inBlockIdx * ELTS_PER_VEC;
                        // Cast float scale to dst data type.
                        using TScale = typename mmha::kv_cache_scale_type_t<T, TCache>::Type;
                        TScale scaleOrigQuant;
                        mmha::convert_from_float(&scaleOrigQuant, params.kvScaleOrigQuant[0]);
                        // Store 8bits kv cache.
                        mmha::store_8bits_vec(kDst, k, inBlockIdx, scaleOrigQuant);
                        mmha::store_8bits_vec(vDst, v, inBlockIdx, scaleOrigQuant);
                    }
                    else
                    {
                        reinterpret_cast<VecT*>(kDst)[inBlockIdx] = k;
                        reinterpret_cast<VecT*>(vDst)[inBlockIdx] = v;
                    }
                }
            }
        }
    }
}

// Use more blocks for the batch dimension in the generation phase.
#define APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, ADD_BIAS, STORE_QKV, DYNAMIC_ROTARY_SCALING, FP8_OUTPUT)               \
    dim3 block(WARP_SIZE, 1);                                                                                          \
    dim3 grid(params.max_input_seq_len, params.head_num);                                                              \
    grid.z = std::min(int(divUp(params.multi_processor_count * WARPS_PER_SM, grid.x * grid.y)),                        \
        int(divUp(params.batch_size, MIN_SEQUENCES_PER_WARP)));                                                        \
    if (params.position_embedding_type == PositionEmbeddingType::kROPE_GPT_NEOX                                        \
        || params.position_embedding_type == PositionEmbeddingType::kLONG_ROPE)                                        \
    {                                                                                                                  \
        applyBiasRopeUpdateKVCache<T, TCache, Dh_MAX, ADD_BIAS, STORE_QKV, KVCacheBuffer,                              \
            RotaryPositionEmbeddingType::GPT_NEOX, DYNAMIC_ROTARY_SCALING, FP8_OUTPUT>                                 \
            <<<grid, block, 0, stream>>>(params);                                                                      \
    }                                                                                                                  \
    else if (params.position_embedding_type == PositionEmbeddingType::kROPE_GPTJ)                                      \
    {                                                                                                                  \
        applyBiasRopeUpdateKVCache<T, TCache, Dh_MAX, ADD_BIAS, STORE_QKV, KVCacheBuffer,                              \
            RotaryPositionEmbeddingType::GPTJ, DYNAMIC_ROTARY_SCALING, FP8_OUTPUT>                                     \
            <<<grid, block, 0, stream>>>(params);                                                                      \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        applyBiasRopeUpdateKVCache<T, TCache, Dh_MAX, ADD_BIAS, STORE_QKV, KVCacheBuffer,                              \
            RotaryPositionEmbeddingType::NONE, DYNAMIC_ROTARY_SCALING, FP8_OUTPUT>                                     \
            <<<grid, block, 0, stream>>>(params);                                                                      \
    }

#define DYNAMIC_ROTARY_SCALING_AND_FP8_OUTPUT_DISPATCH(ADD_BIAS, STORE_QKV)                                            \
    if (dynamic_rotary_scaling)                                                                                        \
    {                                                                                                                  \
        if (params.quantized_fp8_output)                                                                               \
        {                                                                                                              \
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, ADD_BIAS, STORE_QKV, true, true);                                  \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, ADD_BIAS, STORE_QKV, true, false);                                 \
        }                                                                                                              \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        if (params.quantized_fp8_output)                                                                               \
        {                                                                                                              \
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, ADD_BIAS, STORE_QKV, false, true);                                 \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE(Dh_MAX, ADD_BIAS, STORE_QKV, false, false);                                \
        }                                                                                                              \
    }

template <int Dh_MAX, typename T, typename TCache, typename KVCacheBuffer>
void kernelDispatchHeadSize(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream)
{
    bool const add_bias = params.qkv_bias != nullptr;
    bool const store_packed_qkv = !params.separate_q_kv_output;
    bool const dynamic_rotary_scaling = params.rotary_scale_type == RotaryScalingType::kDYNAMIC
        && params.max_input_seq_len > params.rotary_embedding_max_positions;

    constexpr int VEC_SIZE = Rotary_vec_t<T, Dh_MAX>::size;
    // Make sure we have multiple of paired vectors so that the access is aligned.
    TLLM_CHECK_WITH_INFO((params.position_embedding_type != PositionEmbeddingType::kROPE_GPT_NEOX
                             && params.position_embedding_type != PositionEmbeddingType::kLONG_ROPE)
            || params.half_rotary_dim % VEC_SIZE == 0,
        "Rotary dim size is not supported.");

    if (add_bias)
    {
        if (store_packed_qkv)
        {
            DYNAMIC_ROTARY_SCALING_AND_FP8_OUTPUT_DISPATCH(true, true);
        }
        else
        {
            DYNAMIC_ROTARY_SCALING_AND_FP8_OUTPUT_DISPATCH(true, false);
        }
    }
    else
    {
        if (store_packed_qkv)
        {
            DYNAMIC_ROTARY_SCALING_AND_FP8_OUTPUT_DISPATCH(false, true);
        }
        else
        {
            DYNAMIC_ROTARY_SCALING_AND_FP8_OUTPUT_DISPATCH(false, false);
        }
    }
}

template <typename T, typename TCache, typename KVCacheBuffer>
void kernelV1Dispatch(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream)
{
    // Fall back to v1 kernel.
    // GPTJ Rotary embedding needs at least two elements per thread.
    if (params.size_per_head <= 64)
    {
        kernelDispatchHeadSize<64, T, TCache, KVCacheBuffer>(params, stream);
    }
    else if (params.size_per_head <= 128)
    {
        kernelDispatchHeadSize<128, T, TCache, KVCacheBuffer>(params, stream);
    }
    else if (params.size_per_head <= 256)
    {
        kernelDispatchHeadSize<256, T, TCache, KVCacheBuffer>(params, stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            false, "applyBiasRopeUpdateKVCache kernel doesn't support head size = %d", params.size_per_head);
    }
}

#define APPLY_BIAS_ROPE_UPDATE_KV_CACHE_V2(ADD_BIAS, STORE_QKV, FP8_OUTPUT)                                            \
    dim3 block(BLOCK_SIZE);                                                                                            \
    dim3 grid(int(divUp(params.max_input_seq_len, tokens_per_cuda_block)), params.batch_size, params.head_num);        \
    if (params.position_embedding_type == PositionEmbeddingType::kROPE_GPT_NEOX                                        \
        || params.position_embedding_type == PositionEmbeddingType::kLONG_ROPE)                                        \
    {                                                                                                                  \
        applyBiasRopeUpdateKVCacheV2<T, TCache, BLOCK_SIZE, Dh, ADD_BIAS, STORE_QKV, FP8_OUTPUT, KVCacheBuffer,        \
            RotaryPositionEmbeddingType::GPT_NEOX><<<grid, block, 0, stream>>>(params);                                \
    }                                                                                                                  \
    else if (params.position_embedding_type == PositionEmbeddingType::kROPE_GPTJ)                                      \
    {                                                                                                                  \
        applyBiasRopeUpdateKVCacheV2<T, TCache, BLOCK_SIZE, Dh, ADD_BIAS, STORE_QKV, FP8_OUTPUT, KVCacheBuffer,        \
            RotaryPositionEmbeddingType::GPTJ><<<grid, block, 0, stream>>>(params);                                    \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        applyBiasRopeUpdateKVCacheV2<T, TCache, BLOCK_SIZE, Dh, ADD_BIAS, STORE_QKV, FP8_OUTPUT, KVCacheBuffer,        \
            RotaryPositionEmbeddingType::NONE><<<grid, block, 0, stream>>>(params);                                    \
    }

#define STORE_QKV_AND_FP8_OUTPUT_DISPATCH(ADD_BIAS)                                                                    \
    if (store_packed_qkv)                                                                                              \
    {                                                                                                                  \
        if (params.quantized_fp8_output)                                                                               \
        {                                                                                                              \
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE_V2(ADD_BIAS, true, true);                                                  \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE_V2(ADD_BIAS, true, false);                                                 \
        }                                                                                                              \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        if (params.quantized_fp8_output)                                                                               \
        {                                                                                                              \
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE_V2(ADD_BIAS, false, true);                                                 \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            APPLY_BIAS_ROPE_UPDATE_KV_CACHE_V2(ADD_BIAS, false, false);                                                \
        }                                                                                                              \
    }

template <int BLOCK_SIZE, int Dh, typename T, typename TCache, typename KVCacheBuffer>
void kernelV2DispatchHeadSize(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream)
{
    bool const add_bias = params.qkv_bias != nullptr;
    bool const store_packed_qkv = !params.separate_q_kv_output;
    int const vecs_per_head = (params.size_per_head * sizeof(T) / 16);
    TLLM_CHECK_WITH_INFO(BLOCK_SIZE % vecs_per_head == 0, "Kernel block should be able to handle entire heads.");
    int const tokens_per_cuda_block = BLOCK_SIZE / vecs_per_head;

    if (add_bias)
    {
        STORE_QKV_AND_FP8_OUTPUT_DISPATCH(true);
    }
    else
    {
        STORE_QKV_AND_FP8_OUTPUT_DISPATCH(false);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename TCache, typename KVCacheBuffer, int BLOCK_SIZE, int Dh, bool FP8_OUTPUT>
__global__ void updateKVCacheForCrossAttention(QKVPreprocessingParams<T, KVCacheBuffer> params)
{
    // For cross-attention,
    // 1. Load Q from params.qkv_input, and store it to params.q_output.
    // 2. Load K,V from params.cross_kv_input, and store it to params.kv_cache_buffer.

    // NOTE:
    // head_num == kv_head_num
    //   QKV src shape (num_tokens, 3, head_num, size_per_head)
    //                  ^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                           m                        n
    // head_num != kv_head_num
    //   QKV src shape: (num_tokens, head_num * size_per_head + 2 * kv_head_num, size_per_head)
    //                   ^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                             m                               n
    // Q dst shape: (num_tokens, head_num, size_per_head)
    // KV dst shape: refer to the kvCacheBuffer.

    // Constants.
    using VecT = typename VecType<T>::Type;
    constexpr int ELTS_PER_VEC = /* 16bytes */ 16 / sizeof(T);
    constexpr int VECS_PER_HEAD = Dh / ELTS_PER_VEC;
    // One thread process one vector per step.
    constexpr int TOKENS_PER_BLOCK = BLOCK_SIZE / VECS_PER_HEAD;
    // Manually tune the block size to make sure all threads have work to do.
    static_assert(BLOCK_SIZE % VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");

    // The head idx.
    int const head_idx = blockIdx.y;
    // The batch idx.
    int const batch_idx = blockIdx.z;

    // The decoder sequence length.
    int const decoder_seq_len = params.seq_lens[batch_idx];
    // The decoder sequence offset.
    int const decoder_seq_offset = params.cu_seq_lens[batch_idx];
    // The decoder cache sequence length (includes the current input).
    int const decoder_cache_seq_len = params.cache_seq_lens[batch_idx];
    // The encoder sequence length.
    int const encoder_seq_len = params.encoder_seq_lens[batch_idx];
    // The encoder sequence offset.
    int const encoder_seq_offset = params.cu_kv_seq_lens[batch_idx];
    // THe maximum sequence length of encoder and decoder.
    int const max_seq_len = max(decoder_seq_len, encoder_seq_len);

    // Only the first chunk needs to store encoder kv input to the kv cache.
    bool const store_encoder_kv_cache = (decoder_seq_len == decoder_cache_seq_len);

    // Offsets and strides.
    int const head_dim_vec_idx = (threadIdx.x % VECS_PER_HEAD);
    int const head_dim_idx = head_dim_vec_idx * ELTS_PER_VEC;
    int const hidden_idx = head_idx * params.size_per_head + head_dim_idx;
    int const kv_head_idx = head_idx / params.qheads_per_kv_head;
    int const hidden_idx_kv = kv_head_idx * params.size_per_head + head_dim_idx;
    int const src_v_offset = params.kv_hidden_size;

    // Cast float scale to dst data type.
    using TScale = typename mmha::kv_cache_scale_type_t<T, TCache>::Type;
    [[maybe_unused]] TScale scale_orig_quant;
    if constexpr (sizeof(TCache) == 1 || FP8_OUTPUT)
    {
        mmha::convert_from_float(&scale_orig_quant, params.kvScaleOrigQuant ? params.kvScaleOrigQuant[0] : 1.0f);
    }

    // For loop in the sequence length dimension.
    // There might be multiple blocks (blockIdx.x) that process the same sequence in order to fully utilize
    for (int token_idx = blockIdx.x * TOKENS_PER_BLOCK + (threadIdx.x / VECS_PER_HEAD); token_idx < max_seq_len;
         token_idx += (gridDim.x * TOKENS_PER_BLOCK))
    {
        // Decoder tokens (i.e. Q tokens).
        if (token_idx < decoder_seq_len)
        {
            // The global token idx in all sequences.
            int global_token_idx = token_idx + decoder_seq_offset;

            // The memory offset.
            auto const src_q_idx = static_cast<size_t>(global_token_idx) * params.hidden_size + hidden_idx;
            auto const dst_q_idx = static_cast<size_t>(global_token_idx) * params.q_hidden_size + hidden_idx;

            // Only load Q tokens from decoder qkv input.
            auto q = *reinterpret_cast<VecT const*>(params.qkv_input + src_q_idx);

            // Quantize the output to fp8.
            if constexpr (FP8_OUTPUT)
            {
                using OutputType = __nv_fp8_e4m3;
                OutputType* quantized_q_ptr = reinterpret_cast<OutputType*>(params.q_output) + dst_q_idx;
                mmha::store_8bits_vec(quantized_q_ptr, q, 0, scale_orig_quant);
            }
            else
            {
                // Store it to a separate q output.
                *reinterpret_cast<VecT*>(params.q_output + dst_q_idx) = q;
            }
        }

        // Encoder tokens (i.e. KV tokens).
        if (head_idx == (kv_head_idx * params.qheads_per_kv_head) && token_idx < encoder_seq_len
            && store_encoder_kv_cache)
        {
            // The global token idx in all sequences.
            int global_token_idx = token_idx + encoder_seq_offset;

            // The memory offset.
            auto const src_k_idx = static_cast<size_t>(global_token_idx) * params.kv_hidden_size * 2 + hidden_idx_kv;
            auto const src_v_idx
                = static_cast<size_t>(global_token_idx) * params.kv_hidden_size * 2 + src_v_offset + hidden_idx_kv;

            // Only load K,V tokens from encoder qkv input.
            auto k = *reinterpret_cast<VecT const*>(&params.cross_kv_input[src_k_idx]);
            auto v = *reinterpret_cast<VecT const*>(&params.cross_kv_input[src_v_idx]);

            // The kv cache pointers.
            auto k_cache_block_ptr
                = reinterpret_cast<TCache*>(params.kv_cache_buffer.getKBlockPtr(batch_idx, token_idx));
            auto v_cache_block_ptr
                = reinterpret_cast<TCache*>(params.kv_cache_buffer.getVBlockPtr(batch_idx, token_idx));
            // The vector idx in the cache block.
            auto block_vec_idx
                = params.kv_cache_buffer.getKVLocalIdx(token_idx, kv_head_idx, VECS_PER_HEAD, head_dim_vec_idx);

            // Store K and V to the cache.
            // INT8/FP8 kv cache.
            if constexpr (sizeof(TCache) == 1)
            {
                // The element index inside the block.
                auto block_elt_idx = block_vec_idx * ELTS_PER_VEC;
                // Store 8bits kv cache.
                mmha::store_8bits_vec(k_cache_block_ptr, k, block_elt_idx, scale_orig_quant);
                mmha::store_8bits_vec(v_cache_block_ptr, v, block_elt_idx, scale_orig_quant);
            }
            else
            {
                reinterpret_cast<VecT*>(k_cache_block_ptr)[block_vec_idx] = k;
                reinterpret_cast<VecT*>(v_cache_block_ptr)[block_vec_idx] = v;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <int BLOCK_SIZE, int Dh, typename T, typename TCache, typename KVCacheBuffer>
void invokeUpdateKvCacheForCrossAttention(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream)
{
    constexpr int VECS_PER_HEAD = (Dh * sizeof(T) / 16);
    constexpr int TOKENS_PER_CUDA_BLOCK = BLOCK_SIZE / VECS_PER_HEAD;
    TLLM_CHECK_WITH_INFO(BLOCK_SIZE % VECS_PER_HEAD == 0, "Kernel block should be able to handle entire heads.");

    // The grid, block size for kernel launch.
    dim3 block(BLOCK_SIZE);

    // The maximum sequence length of encoder and decoder inputs.
    int const max_seq_len = std::max(params.max_input_seq_len, params.max_kv_seq_len);
    // Assume each SM can hold 2048 threads.
    int const num_blocks_per_sm = 2048 / BLOCK_SIZE;
    // Use more blocks for the token dimension if possible.
    int num_seq_blocks
        = int(divUp(params.multi_processor_count * num_blocks_per_sm, params.head_num * params.batch_size));
    // Make sure we don't launch too many blocks which have no work to do.
    num_seq_blocks = std::min(num_seq_blocks, int(divUp(max_seq_len, TOKENS_PER_CUDA_BLOCK)));
    // The final grid dimension.
    dim3 grid(num_seq_blocks, params.head_num, params.batch_size);

    // Launch the kernel.
    if (params.quantized_fp8_output)
    {
        updateKVCacheForCrossAttention<T, TCache, KVCacheBuffer, BLOCK_SIZE, Dh, true>
            <<<grid, block, 0, stream>>>(params);
    }
    else
    {
        updateKVCacheForCrossAttention<T, TCache, KVCacheBuffer, BLOCK_SIZE, Dh, false>
            <<<grid, block, 0, stream>>>(params);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename TCache, typename KVCacheBuffer>
void invokeApplyBiasRopeUpdateKVCacheDispatch(QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream)
{
    // Use specialized kernels for different heads (better balance of work).
    TLLM_CHECK_WITH_INFO(params.size_per_head % 8 == 0, "Head size needs to be multiple of 8!");
    TLLM_CHECK_WITH_INFO(params.rotary_embedding_dim % 8 == 0, "Rotary embedding dimension needs to be multiple of 8!");
    TLLM_CHECK_WITH_INFO(
        !(params.quantized_fp8_output && !params.separate_q_kv_output && params.quantized_qkv_output == nullptr)
            && !(params.quantized_fp8_output && params.separate_q_kv_output && params.q_output == nullptr),
        "Separate quantized buffer is not provided!");
    bool const absolute_position_embedding
        = (params.position_embedding_type == PositionEmbeddingType::kLEARNED_ABSOLUTE);

    // Launch kernels for cross-attention.
    if (params.cross_attention)
    {
        TLLM_CHECK_WITH_INFO((absolute_position_embedding && params.remove_padding && params.qkv_bias == nullptr),
            "Assume cross attention has learned_absolute position embedding, remove_padding is enabled and no bias");
        switch (params.size_per_head)
        {
        case 32: invokeUpdateKvCacheForCrossAttention<1024, 32, T, TCache, KVCacheBuffer>(params, stream); break;
        case 64: invokeUpdateKvCacheForCrossAttention<1024, 64, T, TCache, KVCacheBuffer>(params, stream); break;
        case 128: invokeUpdateKvCacheForCrossAttention<1024, 128, T, TCache, KVCacheBuffer>(params, stream); break;
        case 256: invokeUpdateKvCacheForCrossAttention<1024, 256, T, TCache, KVCacheBuffer>(params, stream); break;
        default: TLLM_CHECK_WITH_INFO(false, "Not supported."); break;
        }
        return;
    }

    // Long-sequence-length that exceeds the max_position_size needs to compute the cos/sin on-the-fly.
    bool const long_seq_rotary_support = params.rotary_scale_type == RotaryScalingType::kDYNAMIC
        || params.max_kv_seq_len > params.rotary_embedding_max_positions;
    bool const has_rotary_cos_sin_cache = params.rotary_coef_cache_buffer != nullptr;
    bool const has_sink_tokens = params.sink_token_len > 0;
    // V2 implementation requires multiple of paired 16 bytes for gpt-neox rotation.
    bool const support_rotary_for_v2 = (params.position_embedding_type != PositionEmbeddingType::kROPE_GPT_NEOX
                                           && params.position_embedding_type != PositionEmbeddingType::kLONG_ROPE)
        || params.rotary_embedding_dim % 16 == 0;

    // Use v2 kernel for absolute_position_embedding.
    if (!absolute_position_embedding
        && (long_seq_rotary_support || !has_rotary_cos_sin_cache || has_sink_tokens || !support_rotary_for_v2))
    {
        kernelV1Dispatch<T, TCache, KVCacheBuffer>(params, stream);
        return;
    }

    // Attempt to adopt optimized memory-bound kernel first, otherwise fall back to the v1 kernel.
    switch (params.size_per_head)
    {
    case 32: kernelV2DispatchHeadSize<256, 32, T, TCache, KVCacheBuffer>(params, stream); break;
    case 48: kernelV2DispatchHeadSize<192, 48, T, TCache, KVCacheBuffer>(params, stream); break;
    case 64: kernelV2DispatchHeadSize<256, 64, T, TCache, KVCacheBuffer>(params, stream); break;
    case 80: kernelV2DispatchHeadSize<160, 80, T, TCache, KVCacheBuffer>(params, stream); break;
    case 96: kernelV2DispatchHeadSize<192, 96, T, TCache, KVCacheBuffer>(params, stream); break;
    case 104: kernelV2DispatchHeadSize<416, 104, T, TCache, KVCacheBuffer>(params, stream); break;
    case 112: kernelV2DispatchHeadSize<224, 112, T, TCache, KVCacheBuffer>(params, stream); break;
    case 128: kernelV2DispatchHeadSize<256, 128, T, TCache, KVCacheBuffer>(params, stream); break;
    case 144: kernelV2DispatchHeadSize<288, 144, T, TCache, KVCacheBuffer>(params, stream); break;
    case 160: kernelV2DispatchHeadSize<160, 160, T, TCache, KVCacheBuffer>(params, stream); break;
    case 192: kernelV2DispatchHeadSize<192, 192, T, TCache, KVCacheBuffer>(params, stream); break;
    case 224: kernelV2DispatchHeadSize<224, 224, T, TCache, KVCacheBuffer>(params, stream); break;
    case 256: kernelV2DispatchHeadSize<256, 256, T, TCache, KVCacheBuffer>(params, stream); break;
    default:
        // Fall back to v1 kernel.
        // GPTJ Rotary embedding needs at least two elements per thread.
        kernelV1Dispatch<T, TCache, KVCacheBuffer>(params, stream);
        break;
    }
}

#define INSTANTIATE_ADDFUSEDQKVBIAS_TRANSPOSE(T, TCache, KVCacheBuffer)                                                \
    template void invokeApplyBiasRopeUpdateKVCacheDispatch<T, TCache, KVCacheBuffer>(                                  \
        QKVPreprocessingParams<T, KVCacheBuffer> params, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
