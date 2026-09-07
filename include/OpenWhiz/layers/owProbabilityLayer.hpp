/*
 * owProbabilityLayer.hpp
 *
 *  Created on: Jan 12, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owLayer.hpp"
#include <cmath>
#include <algorithm>

namespace ow {

/**
 * @class owProbabilityLayer
 * @brief Implements the Softmax activation function to convert logits into probabilities.
 * 
 * This layer takes a vector of real numbers and normalizes it into a probability distribution, 
 * where each value is between 0 and 1 and the sum of all values is 1.
 * 
 * @details
 * **Implementation Details:**
 * - Uses the "Log-Sum-Exp" trick by subtracting the maximum value before exponentiation to 
 *   ensure numerical stability.
 * - Softmax formula: sigma(z)_i = exp(z_i - max(z)) / sum(exp(z_j - max(z))).
 * - Supports gradient backpropagation, which is particularly efficient when paired with 
 *   Cross-Entropy loss.
 * 
 * **Unique Features:**
 * - Numerical stability through max-subtraction.
 * - Essential for multi-class classification tasks.
 * 
 * **Platform-Specific Notes:**
 * - **Computer/Mobile/Web:** Performance is dominated by exponential functions; optimized math 
 *   libraries can provide significant speedups.
 * - **Industrial:** Useful for decision-making systems where the model needs to provide confidence 
 *   levels for different diagnostic states.
 */
class owProbabilityLayer : public owLayer {
public:
    /**
     * @brief Constructs a Probability (Softmax) layer.
     */
    owProbabilityLayer() { m_layerName = "Probability Layer"; }
    size_t getInputSize() const override { return m_lastOutput.shape()[1]; }
    size_t getOutputSize() const override { return m_lastOutput.shape()[1]; }
    void setNeuronNum(size_t num) override { m_lastOutput = owTensor<float, 2>(1, num); }
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owProbabilityLayer>();
        copy->m_layerName = m_layerName;
        return copy;
    }
    std::string toXML() const override { return "<Info>Softmax</Info>\n"; }
    void fromXML(const std::string& xml) override {}
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        owTensor<float, 2> output(input.shape());
        for (size_t b = 0; b < input.shape()[0]; ++b) {
            float maxVal = input(b, 0);
            for (size_t f = 1; f < input.shape()[1]; ++f) maxVal = std::max(maxVal, input(b, f));
            float sum = 0.0f;
            for (size_t f = 0; f < input.shape()[1]; ++f) { output(b, f) = std::exp(input(b, f) - maxVal); sum += output(b, f); }
            // Clamp to the same eps as owCategoricalCrossEntropyLoss's prediction
            // clamp, so a confidently-wrong logit can't underflow past it and
            // decouple the two gradients toward zero.
            const float eps = 1e-12f;
            for (size_t f = 0; f < input.shape()[1]; ++f) {
                output(b, f) = std::max(eps, std::min(1.0f - eps, output(b, f) / sum));
            }
        }
        m_lastOutput = output; return output;
    }
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        owTensor<float, 2> inputGradient(outputGradient.shape());
        for (size_t b = 0; b < outputGradient.shape()[0]; ++b) {
            for (size_t i = 0; i < outputGradient.shape()[1]; ++i) {
                float dotProd = 0.0f;
                for (size_t j = 0; j < outputGradient.shape()[1]; ++j) dotProd += outputGradient(b, j) * m_lastOutput(b, j);
                inputGradient(b, i) = m_lastOutput(b, i) * (outputGradient(b, i) - dotProd);
            }
        }
        return inputGradient;
    }
    void train() override {}

	float* getParamsPtr() override { return nullptr; }
	float* getGradsPtr() override { return nullptr; }
	size_t getParamsCount() override { return 0; }

private:
    owTensor<float, 2> m_lastOutput;
};

} // namespace ow
