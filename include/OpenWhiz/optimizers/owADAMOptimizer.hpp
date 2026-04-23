/*
 * owADAMOptimizer.hpp
 *
 *  Created on: Jan 12, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owOptimizer.hpp"
#include "../core/owCuda.hpp"

namespace ow {

/**
 * @class owADAMOptimizer
 * @brief Adaptive Moment Estimation (Adam) optimizer.
 * 
 * Adam is currently the most widely used optimizer in deep learning. It combines 
 * the benefits of both Momentum (first moment) and RMSProp (second moment) 
 * to provide a robust and fast-converging optimization.
 * 
 * Formula:
 * m_t = β1 * m_{t-1} + (1 - β1) * g_t
 * v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
 * m_hat = m_t / (1 - β1^t)
 * v_hat = v_t / (1 - β2^t)
 * θ = θ - (η / (sqrt(v_hat) + ε)) * m_hat
 * 
 * **Advantages:**
 * - **Fastest Convergence:** Usually reaches a minimum in fewer iterations.
 * - **Robust Hyperparameters:** Default values often work well across many tasks.
 * - **Handles Sparse Gradients:** Performs well even with irregular data updates.
 * 
 * **Memory Considerations:**
 * - Requires **two additional memory buffers** per parameter (m and v). 
 * - In **mobile and memory-constrained web apps**, this can double or triple the 
 *   memory footprint of a large model compared to SGD.
 */
class owADAMOptimizer : public owOptimizer {
public:
    /**
     * @brief Constructs an Adam optimizer.
     * @param lr The initial learning rate (default: 0.001).
     * @param b1 The decay rate for the first moment (default: 0.9).
     * @param b2 The decay rate for the second moment (default: 0.999).
     * @param eps A small constant for numerical stability (default: 1e-8).
     */
    owADAMOptimizer(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : m_beta1(b1), m_beta2(b2), m_epsilon(eps) { m_learningRate = lr; }

    /**
     * @brief Performs an Adam update step.
     * 
     * Computes bias-corrected first and second moment estimates and applies them 
     * to the parameters.
     *
     * @param params The parameter tensor to update.
     * @param gradients The gradient tensor.
     */
    void update(owTensor<float, 2>& params, const owTensor<float, 2>& gradients) override {
        auto g_clipped = clipGradients(gradients);
        auto& m = getBuffer(&params, params.shape(), 0);
        auto& v = getBuffer(&params, params.shape(), 1);
        m_t++;

#ifdef OW_USE_GPU
        cuda::adamUpdate(params.data(), g_clipped.data(), m.data(), v.data(), 
                        (int)params.size(), m_learningRate, m_beta1, m_beta2, m_epsilon, m_t);
#else
        float b1t = std::pow(m_beta1, m_t);
        float b2t = std::pow(m_beta2, m_t);

        for (size_t i = 0; i < params.size(); ++i) {
            float g = g_clipped.data()[i];
            m.data()[i] = m_beta1 * m.data()[i] + (1.0f - m_beta1) * g;
            v.data()[i] = m_beta2 * v.data()[i] + (1.0f - m_beta2) * g * g;

            float m_hat = m.data()[i] / (1.0f - b1t);
            float v_hat = v.data()[i] / (1.0f - b2t);

            params.data()[i] -= m_learningRate * m_hat / (std::sqrt(v_hat) + m_epsilon);
        }
#endif
    }

    /**
     * @brief Returns the name of the optimizer.
     */
    std::string getOptimizerName() const override { return "ADAM"; }

    /**
     * @brief Creates a deep copy of the Adam optimizer.
     * @return std::shared_ptr<owOptimizer> A shared pointer to the cloned instance.
     */
    std::shared_ptr<owOptimizer> clone() const override {
        auto copy = std::make_shared<owADAMOptimizer>(m_learningRate, m_beta1, m_beta2, m_epsilon);
        copy->m_gradientClipThreshold = m_gradientClipThreshold;
        copy->m_t = m_t;
        copyBuffersTo(copy.get());
        return copy;
    }
private:
    float m_beta1, m_beta2, m_epsilon;
    int m_t = 0;
};

} // namespace ow
