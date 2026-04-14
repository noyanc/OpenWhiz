/*
 * owSequentialLayer.hpp
 *
 *  Created on: Apr 12, 2026
 *      Author: Noyan Culum, AITIAL
 */

#pragma once
#include "owLayer.hpp"
#include <vector>
#include <memory>

namespace ow {

/**
 * @examples\classificationExample\classification_data.csv owSequentialLayer
 * @brief A container layer that executes a sequence of layers internally.
 * Supports Independent Branch Training via local loss calculation and gradient injection.
 */
class owSequentialLayer : public owLayer {
public:
    owSequentialLayer() {
        m_layerName = "Sequential Layer";
    }

    void addLayer(std::shared_ptr<owLayer> layer) {
        if (layer) m_layers.push_back(layer);
    }

    size_t getInputSize() const override {
        return m_layers.empty() ? 0 : m_layers.front()->getInputSize();
    }

    size_t getOutputSize() const override {
        return m_layers.empty() ? 0 : m_layers.back()->getOutputSize();
    }

    void setNeuronNum(size_t num) override {
        if (!m_layers.empty()) m_layers.front()->setNeuronNum(num);
    }

    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        owTensor<float, 2> output = input;
        for (auto& layer : m_layers) output = layer->forward(output);
        m_lastOutput = output; // Cache for local loss
        return output;
    }

    /**
     * @brief Performs backward pass, optionally injecting local gradients for expert training.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        owTensor<float, 2> grad = outputGradient;
        
        // --- Independent Expertise Injection ---
        // Only inject local gradients if this layer is in Independent Expert Mode!
        if (m_isIndependentExpertMode && m_localTarget && !m_isFrozen && m_lastOutput.size() > 0) {
            size_t batchSize = m_lastOutput.shape()[0];
            
            // Assuming MSE loss for local expertise: L = 0.5 * (out - target)^2
            // dL/dout = (out - target) * weight
            if (m_localTarget->size() == m_lastOutput.size()) {
                for (size_t i = 0; i < grad.size(); ++i) {
                    float localGrad = (m_lastOutput.data()[i] - m_localTarget->data()[i]) / (float)batchSize;
                    grad.data()[i] += localGrad * m_localExpertWeight;
                }
            }
        }

        // Propagate combined gradient through internal layers
        for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
            grad = (*it)->backward(grad);
        }
        return grad;
    }

    /**
     * @brief Computes MSE between internal output and local target.
     */
    float computeLocalLoss(const owTensor<float, 2>& target) {
        if (m_lastOutput.size() == 0 || target.size() == 0 || target.size() != m_lastOutput.size()) return 1e30f;
        float mse = 0;
        size_t n = m_lastOutput.size();
        for (size_t i = 0; i < n; ++i) {
            float diff = m_lastOutput.data()[i] - target.data()[i];
            mse += diff * diff;
        }
        return mse / (float)n;
    }

    /**
     * @brief Manually triggers a training step using only the local target.
     * Only works if Independent Expert Mode is enabled.
     */
    void trainIndependentExpertOnly() {
        if (!m_isIndependentExpertMode || !m_localTarget || m_isFrozen || m_lastOutput.size() == 0) return;
        
        size_t batchSize = m_lastOutput.shape()[0];
        owTensor<float, 2> grad(m_lastOutput.shape()[0], m_lastOutput.shape()[1]);
        
        for (size_t i = 0; i < grad.size(); ++i) {
            grad.data()[i] = (m_lastOutput.data()[i] - m_localTarget->data()[i]) / (float)batchSize;
        }

        backward(grad);
        train();
    }

    void train() override {
        if (m_isFrozen) return; // Independent stopping!
        for (auto& layer : m_layers) layer->train();
    }

    void reset() override {
        for (auto& layer : m_layers) layer->reset();
    }

    void setOptimizer(owOptimizer* opt) override {
        owLayer::setOptimizer(opt);
        for (auto& layer : m_layers) layer->setOptimizer(opt);
    }

    void setRegularization(int type) override {
        owLayer::setRegularization(type);
        for (auto& layer : m_layers) layer->setRegularization(type);
    }

    float* getParamsPtr() override { return nullptr; } 
    float* getGradsPtr() override { return nullptr; }
    size_t getParamsCount() override {
        size_t total = 0;
        for (const auto& l : m_layers) total += l->getParamsCount();
        return total;
    }

    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owSequentialLayer>();
        for (const auto& l : m_layers) copy->addLayer(l->clone());
        copy->m_isFrozen = m_isFrozen;
        copy->m_convergenceThreshold = m_convergenceThreshold;
        copy->m_isIndependentExpertMode = m_isIndependentExpertMode;
        return copy;
    }

    std::string toXML() const override {
        std::stringstream ss;
        ss << "<LayerCount>" << m_layers.size() << "</LayerCount>\n";
        ss << "<Frozen>" << (m_isFrozen ? 1 : 0) << "</Frozen>\n";
        ss << "<IndependentExpertMode>" << (m_isIndependentExpertMode ? 1 : 0) << "</IndependentExpertMode>\n";
        ss << "<Threshold>" << m_convergenceThreshold << "</Threshold>\n";
        for (size_t i = 0; i < m_layers.size(); ++i) {
            ss << "<SubLayer_" << i << " type=\"" << m_layers[i]->getLayerName() << "\">\n"
               << m_layers[i]->toXML() << "</SubLayer_" << i << ">\n";
        }
        return ss.str();
    }

    void fromXML(const std::string& xml) override {
        m_isFrozen = (getTagContent(xml, "Frozen") == "1");
        m_isIndependentExpertMode = (getTagContent(xml, "IndependentExpertMode") == "1");
        std::string thr = getTagContent(xml, "Threshold");
        if (!thr.empty()) m_convergenceThreshold = std::stof(thr);
        for (size_t i = 0; i < m_layers.size(); ++i) {
            std::string tag = "SubLayer_" + std::to_string(i);
            m_layers[i]->fromXML(owLayer::getNestedTagContent(xml, tag));
        }
    }

private:
    std::vector<std::shared_ptr<owLayer>> m_layers;
    owTensor<float, 2> m_lastOutput;
};

} // namespace ow
