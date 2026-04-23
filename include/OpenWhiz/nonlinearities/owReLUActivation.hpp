/*
 * owReLUActivation.hpp
 *
 *  Created on: Dec 16, 2025
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owActivation.hpp"
#include "../core/owSimd.hpp"
#include "../core/owCuda.hpp"

namespace ow {

/**
 * @class owReLUActivation
 * @brief Implements the Rectified Linear Unit (ReLU) activation function.
 * 
 * ReLU is the most popular activation function for deep neural networks. It returns 0 
 * for any negative input and the input itself for any positive input (f(x) = max(0, x)).
 * 
 * Performance characteristics:
 * - Extremely fast to compute compared to Sigmoid or Tanh.
 * - Promotes sparsity in neural networks (many neurons output exactly zero).
 * - Optimized with AVX2 SIMD instructions in OpenWhiz for high-throughput server and computer applications.
 * 
 * Warning: Standard ReLU is susceptible to the "dying ReLU" problem. For very deep networks 
 * or sensitive industrial data, consider using owLeakyReLUActivation.
 */
class owReLUActivation : public owActivation {
public:
    /**
     * @brief Forward pass: computes f(x) = max(0, x) element-wise.
     * @param input Input tensor.
     * @return Output tensor with ReLU applied.
     * 
     * This implementation uses AVX2 SIMD optimizations where available to maximize 
     * processing speed on modern CPUs.
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        owTensor<float, 2> out = input;
        float* data = out.data();
        size_t n = out.size();

#ifdef OW_USE_GPU
        cuda::reluForward(data, (int)n);
#else
        #ifdef __AVX2__
        __m256 zero = _mm256_setzero_ps();
        for (size_t i = 0; i <= n - 8; i += 8) {
            _mm256_storeu_ps(data + i, _mm256_max_ps(_mm256_loadu_ps(data + i), zero));
        }
        for (size_t i = (n / 8) * 8; i < n; ++i) data[i] = std::max(0.0f, data[i]);
        #elif defined(OW_ARM_NEON)
        float32x4_t zero = vdupq_n_f32(0.0f);
        for (size_t i = 0; i <= n - 4; i += 4) {
            vst1q_f32(data + i, vmaxq_f32(vld1q_f32(data + i), zero));
        }
        for (size_t i = (n / 4) * 4; i < n; ++i) data[i] = std::max(0.0f, data[i]);
        #else
        for (size_t i = 0; i < n; ++i) data[i] = std::max(0.0f, data[i]);
        #endif
#endif
        return out;
    }

    owTensor<float, 2> backward(const owTensor<float, 2>& input, const owTensor<float, 2>& outputGradient) override {
        owTensor<float, 2> grad = outputGradient;
        float* gData = grad.data();
        const float* iData = input.data();
        size_t n = grad.size();

#ifdef OW_USE_GPU
        cuda::reluBackward(gData, iData, (int)n);
#else
        #ifdef __AVX2__
        __m256 zero = _mm256_setzero_ps();
        for (size_t i = 0; i <= n - 8; i += 8) {
            __m256 mask = _mm256_cmp_ps(_mm256_loadu_ps(iData + i), zero, _CMP_GT_OQ);
            _mm256_storeu_ps(gData + i, _mm256_and_ps(_mm256_loadu_ps(gData + i), mask));
        }
        for (size_t i = (n / 8) * 8; i < n; ++i) gData[i] *= (iData[i] > 0.0f ? 1.0f : 0.0f);
        #elif defined(OW_ARM_NEON)
        float32x4_t zero = vdupq_n_f32(0.0f);
        for (size_t i = 0; i <= n - 4; i += 4) {
            uint32x4_t mask = vcgtq_f32(vld1q_f32(iData + i), zero);
            vst1q_f32(gData + i, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vld1q_f32(gData + i)), mask)));
        }
        for (size_t i = (n / 4) * 4; i < n; ++i) gData[i] *= (iData[i] > 0 ? 1.0f : 0.0f);
        #else
        for (size_t i = 0; i < n; ++i) gData[i] *= (iData[i] > 0 ? 1.0f : 0.0f);
        #endif
#endif
        return grad;
    }

    /**
     * @brief Deep copy of the ReLU instance.
     * @return Shared pointer to new owReLUActivation instance.
     */
    std::shared_ptr<owActivation> clone() const override { return std::make_shared<owReLUActivation>(); }
};

} // namespace ow
