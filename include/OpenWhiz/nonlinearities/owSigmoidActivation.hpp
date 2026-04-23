/*
 * owSigmoidActivation.hpp
 *
 *  Created on: Feb 16, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owActivation.hpp"
#include "../core/owSimd.hpp"
#include "../core/owCuda.hpp"

namespace ow {

/**
 * @class owSigmoidActivation
 * @brief Implements the Logistic Sigmoid activation function.
 * 
 * Sigmoid squashes any real-valued input into a range between 0 and 1. It is 
 * defined as f(x) = 1 / (1 + exp(-x)).
 * 
 * Usage scenarios:
 * - Primarily used in the output layer for binary classification problems.
 * - Suitable for models requiring probability-like outputs (web-based click prediction, etc.).
 * 
 * Note for developers:
 * Sigmoid can suffer from "gradient vanishing" in deep networks because its gradient 
 * is very small for large positive or negative inputs. For hidden layers, 
 * owReLUActivation is generally preferred.
 */
class owSigmoidActivation : public owActivation {
public:
    /**
     * @brief Forward pass: computes the sigmoid function element-wise.
     * @param input Input tensor.
     * @return Output tensor with values squashed between 0 and 1.
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        owTensor<float, 2> out = input;
        size_t n = out.size();
#ifdef OW_USE_GPU
        cuda::sigmoidForward(out.data(), (int)n);
#else
        for (size_t i = 0; i < n; ++i) out.data()[i] = 1.0f / (1.0f + std::exp(-out.data()[i]));
#endif
        return out;
    }

    /**
     * @brief Backward pass: computes sigmoid derivative (s * (1 - s)).
     * @param input Original input tensor.
     * @param outputGradient Incoming gradient from the next layer.
     * @return Resulting gradient for optimization.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& input, const owTensor<float, 2>& outputGradient) override {
        owTensor<float, 2> grad = outputGradient;
        float* gData = grad.data();
        const float* iData = input.data();
        size_t n = grad.size();

#ifdef OW_USE_GPU
        cuda::sigmoidBackward(gData, iData, (int)n); 
#else
        #ifdef __AVX2__
        __m256 v_one = _mm256_set1_ps(1.0f);
        for (size_t i = 0; i <= n - 8; i += 8) {
            float s[8];
            for(int j=0; j<8; ++j) s[j] = 1.0f / (1.0f + std::exp(-iData[i+j]));
            __m256 v_s = _mm256_loadu_ps(s);
            __m256 v_grad = _mm256_loadu_ps(gData + i);
            _mm256_storeu_ps(gData + i, _mm256_mul_ps(v_grad, _mm256_mul_ps(v_s, _mm256_sub_ps(v_one, v_s))));
        }
        for (size_t i = (n / 8) * 8; i < n; ++i) {
            float s = 1.0f / (1.0f + std::exp(-iData[i]));
            gData[i] *= s * (1.0f - s);
        }
        #elif defined(OW_ARM_NEON)
        float32x4_t v_one = vdupq_n_f32(1.0f);
        for (size_t i = 0; i <= n - 4; i += 4) {
            float s[4];
            for(int j=0; j<4; ++j) s[j] = 1.0f / (1.0f + std::exp(-iData[i+j]));
            float32x4_t v_s = vld1q_f32(s);
            float32x4_t v_grad = vld1q_f32(gData + i);
            vst1q_f32(gData + i, vmulq_f32(v_grad, vmulq_f32(v_s, vsubq_f32(v_one, v_s))));
        }
        for (size_t i = (n / 4) * 4; i < n; ++i) {
            float s = 1.0f / (1.0f + std::exp(-iData[i]));
            gData[i] *= s * (1.0f - s);
        }
        #else
        for (size_t i = 0; i < n; ++i) {
            float s = 1.0f / (1.0f + std::exp(-iData[i]));
            gData[i] *= s * (1.0f - s);
        }
        #endif
#endif
        return grad;
    }

    /**
     * @brief Deep copy of the Sigmoid activation.
     * @return Shared pointer to new owSigmoidActivation instance.
     */
    std::shared_ptr<owActivation> clone() const override { return std::make_shared<owSigmoidActivation>(); }
};

} // namespace ow
