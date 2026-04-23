/*
 * owMeanSquaredErrorLoss.hpp
 *
 *  Created on: Dec 16, 2025
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLoss.hpp"
#include "../core/owCuda.hpp"

namespace ow {

/**
 * @class owMeanSquaredErrorLoss
 * @brief Computes the Mean Squared Error (MSE) loss.
 * 
 * MSE loss calculates the average of the squares of the differences between 
 * predictions and targets.
 * 
 * @details
 * Mathematical formula:
 * \f$ L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \f$
 * where \f$ y_i \f$ is the target and \f$ \hat{y}_i \f$ is the prediction.
 * 
 * Features:
 * - Penalizes outliers more heavily than MAE due to the squaring operation.
 * - Resulting loss is always non-negative.
 * - Smooth and differentiable everywhere, which aids convergence in optimization.
 * 
 * Comparison with other losses:
 * - Use MSE when you want to heavily penalize large errors (outliers).
 * - Use MAE (Mean Absolute Error) when your data contains many outliers that should not dominate the loss.
 * - Use Huber loss as a robust alternative that behaves like MSE for small errors and MAE for large ones.
 * 
 * Platform notes:
 * - Computer: Standard choice for most regression tasks.
 * - Mobile/Web: Efficient to compute due to simple arithmetic operations.
 * - Industrial: Good for high-precision tasks where small deviations need to be minimized.
 */
class owMeanSquaredErrorLoss : public owLoss {
public:
    owMeanSquaredErrorLoss() {
#ifdef OW_USE_GPU
        cudaMallocManaged(&m_gpuLoss, sizeof(float));
#endif
    }

    ~owMeanSquaredErrorLoss() {
#ifdef OW_USE_GPU
        if (m_gpuLoss) cudaFree(m_gpuLoss);
#endif
    }

    float compute(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        size_t n = prediction.size();
#ifdef OW_USE_GPU
        cuda::mseLoss(prediction.data(), target.data(), m_gpuLoss, (int)n);
        return *m_gpuLoss;
#else
        float loss = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            float diff = prediction.data()[i] - target.data()[i];
            loss += diff * diff;
        }
        return loss / static_cast<float>(n);
#endif
    }

    owTensor<float, 2> gradient(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        owTensor<float, 2> grad(prediction.shape());
        size_t n = prediction.size();
#ifdef OW_USE_GPU
        cuda::mseGradient(prediction.data(), target.data(), grad.data(), (int)n);
#else
        float factor = 2.0f / static_cast<float>(n);
        for (size_t i = 0; i < n; ++i) grad.data()[i] = factor * (prediction.data()[i] - target.data()[i]);
#endif
        return grad;
    }

    std::string getLossName() const override { return "Mean Squared Error Loss"; }

    std::shared_ptr<owLoss> clone() const override { return std::make_shared<owMeanSquaredErrorLoss>(); }

private:
    float* m_gpuLoss = nullptr;
};

} // namespace ow
