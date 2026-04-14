/*
 * owSlidingWindowLayer.hpp
 *
 *  Created on: Apr 11, 2026
 *      Author: Noyan Culum, AITIAL
 */

#pragma once
#include "owLayer.hpp"

namespace ow {

/**
 * @class owSlidingWindowLayer
 * @brief A stateless layer that slices temporal windows from a pre-formatted forecasting dataset.
 * 
 * In the Data-Centric architecture, owDataset::prepareForecastData() prepares a "Master History" 
 * for each row. This layer acts as a "View" or "Slicer" that selects a specific subset of 
 * that history (e.g., a 5-day window from a 22-day master history).
 * 
 * **Advantages:**
 * 1. Stateless: No internal history buffer, making it 100% compatible with Shuffling.
 * 2. Multi-Scale: Multiple branches can slice different window sizes from the same dataset row.
 * 
 * **Input Structure (prepared by Dataset):**
 * `[H_master, H_master-1, ..., H_1, F1, ..., Fn]`
 * 
 * **Output Structure:**
 * `[t-1, t-2, ..., t-windowSize, F1, ..., Fn]`
 */
class owSlidingWindowLayer : public owLayer {
public:
    /**
     * @brief Constructor for owSlidingWindowLayer.
     * @param windowSize The number of historical points to include in the output vector.
     * @param dilation The gap between historical points (1 = consecutive, 2 = every other point).
     * @param masterWindowSize The size of the master history prepared by the dataset.
     * @param includeCurrent If true, the input feature vector (F1...Fn) is appended to the output.
     */
    owSlidingWindowLayer(size_t windowSize = 5, size_t dilation = 1, size_t masterWindowSize = 5, bool includeCurrent = true) 
        : m_windowSize(windowSize), m_dilation(dilation), m_masterWindowSize(masterWindowSize), m_inputFeatures(1), m_includeCurrent(includeCurrent) {
        m_layerName = "Sliding Window Layer";
    }

    /** @return The total expected input size (Master History + Features). */
    size_t getInputSize() const override { return m_masterWindowSize + m_inputFeatures; } 

    /** 
     * @brief Calculates the output vector size.
     * Formula: windowSize + (includeCurrent ? inputFeatures : 0)
     */
    size_t getOutputSize() const override { return m_windowSize + (m_includeCurrent ? m_inputFeatures : 0); }
    
    /** @brief Updates the internal input feature count. */
    void setNeuronNum(size_t num) override { m_inputFeatures = num; }

    /** @brief Stateless layer, reset does nothing. */
    void reset() override {}

    /** @return A deep copy of the layer. */
    std::shared_ptr<owLayer> clone() const override {
        auto copy = std::make_shared<owSlidingWindowLayer>(m_windowSize, m_dilation, m_masterWindowSize, m_includeCurrent);
        copy->m_layerName = m_layerName;
        copy->m_inputFeatures = m_inputFeatures;
        return copy;
    }

    /** 
     * @brief Slices the requested window from the master history.
     */
    owTensor<float, 2> forward(const owTensor<float, 2>& input) override {
        size_t batchSize = input.shape()[0];
        size_t totalCols = input.shape()[1];
        m_inputFeatures = totalCols - m_masterWindowSize; 

        owTensor<float, 2> output(batchSize, getOutputSize());
        
        for (size_t i = 0; i < batchSize; ++i) {
            // 1. Slice History from Master
            // Dataset puts History_1 (t-1) at index m_masterWindowSize - 1
            for (size_t w = 0; w < m_windowSize; ++w) {
                size_t lookbackSteps = (w + 1) * m_dilation;
                if (lookbackSteps <= m_masterWindowSize) {
                    output(i, w) = input(i, m_masterWindowSize - lookbackSteps);
                } else {
                    output(i, w) = 0.0f; 
                }
            }
            
            // 2. Append Features (F1...Fn)
            if (m_includeCurrent) {
                for (size_t f = 0; f < m_inputFeatures; ++f) {
                    output(i, m_windowSize + f) = input(i, m_masterWindowSize + f);
                }
            }
        }
        return output;
    }

    /** 
     * @brief Propagates gradients back to the original features.
     */
    owTensor<float, 2> backward(const owTensor<float, 2>& outputGradient) override {
        size_t batchSize = outputGradient.shape()[0];
        owTensor<float, 2> inputGradient(batchSize, m_masterWindowSize + m_inputFeatures);
        
        for (size_t i = 0; i < batchSize; ++i) {
            // Gradients for historical columns (usually frozen, but mapped for completeness)
            for (size_t w = 0; w < m_windowSize; ++w) {
                size_t lookbackSteps = (w + 1) * m_dilation;
                if (lookbackSteps <= m_masterWindowSize) {
                    inputGradient(i, m_masterWindowSize - lookbackSteps) = outputGradient(i, w);
                }
            }

            // Gradients for current features
            if (m_includeCurrent) {
                for (size_t f = 0; f < m_inputFeatures; ++f) {
                    inputGradient(i, m_masterWindowSize + f) = outputGradient(i, m_windowSize + f);
                }
            }
        }
        return inputGradient;
    }

    /** @return XML formatted configuration string. */
    std::string toXML() const override {
        std::stringstream ss;
        ss << "<WindowSize>" << m_windowSize << "</WindowSize>\n";
        ss << "<Dilation>" << m_dilation << "</Dilation>\n";
        ss << "<MasterWindowSize>" << m_masterWindowSize << "</MasterWindowSize>\n";
        ss << "<InputFeatures>" << m_inputFeatures << "</InputFeatures>\n";
        ss << "<IncludeCurrent>" << (m_includeCurrent ? 1 : 0) << "</IncludeCurrent>\n";
        return ss.str();
    }

    /** @brief Reconstructs from XML. */
    void fromXML(const std::string& xml) override {
        m_windowSize = std::stoul(getTagContent(xml, "WindowSize"));
        m_dilation = std::stoul(getTagContent(xml, "Dilation"));
        m_masterWindowSize = std::stoul(getTagContent(xml, "MasterWindowSize"));
        m_inputFeatures = std::stoul(getTagContent(xml, "InputFeatures"));
        m_includeCurrent = std::stoi(getTagContent(xml, "IncludeCurrent")) != 0;
    }

    void train() override {}
    float* getParamsPtr() override { return nullptr; }
    float* getGradsPtr() override { return nullptr; }
    size_t getParamsCount() override { return 0; }

    void setIncludeCurrent(bool include) { m_includeCurrent = include; }
    void setMasterWindowSize(size_t size) { m_masterWindowSize = size; }

private:
    size_t m_windowSize;    
    size_t m_dilation;      
    size_t m_masterWindowSize; 
    size_t m_inputFeatures; 
    bool m_includeCurrent;  
};

} // namespace ow
