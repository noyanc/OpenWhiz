/*
 * owNeuralNetwork.inl
 *
 *  Created on: Jan 12, 2026
 *      Author: Noyan Culum, AITIAL
 */


#pragma once

namespace ow {

inline owNeuralNetwork::owNeuralNetwork() {
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    m_rng.seed(static_cast<unsigned int>(seed));
    m_dataset = std::make_shared<owDataset>();
    m_optimizer = std::make_shared<owLBFGSOptimizer>(); // Default optimizer
    m_loss = std::make_shared<owMeanSquaredErrorLoss>(); // Default loss
}

inline void owNeuralNetwork::setDataset(std::shared_ptr<owDataset> ds) { m_dataset = ds; }
inline owDataset* owNeuralNetwork::getDataset() { return m_dataset.get(); }
inline bool owNeuralNetwork::loadData(const std::string& filename, bool hasHeader, bool autoNormalize) { return m_dataset->loadFromCSV(filename, hasHeader, autoNormalize); }

inline void owNeuralNetwork::setTraining(bool training) {
    for (auto& layer : m_layers) layer->setTraining(training);
}

inline void owNeuralNetwork::setRegularization(int type) { m_regType = type; for (auto& layer : m_layers) layer->setRegularization(type); }
inline void owNeuralNetwork::addLayer(std::shared_ptr<owLayer> layer) {
    if (layer) { 
        layer->setParentNetwork(this);
        layer->setOptimizer(m_optimizer.get()); 
        layer->setRegularization(m_regType); 

        if (layer->getInputSize() == 0) {
            size_t prevOutput = 0;
            if (m_layers.empty()) {
                if (m_dataset) prevOutput = (size_t)m_dataset->getInputVariableNum();
            } else {
                prevOutput = m_layers.back()->getOutputSize();
            }
            if (prevOutput > 0) layer->setInputSize(prevOutput);
        }

        m_layers.push_back(layer); 
    }
}

inline void owNeuralNetwork::getInputMinMax(owTensor<float, 2>& min, owTensor<float, 2>& max) const {
    if (!m_dataset) return;
    auto indices = m_dataset->getUsedColumnIndices(false);
    min = owTensor<float, 2>(1, indices.size());
    max = owTensor<float, 2>(1, indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        auto params = m_dataset->getNormalizationParamsByColumnIndex(indices[i]);
        min(0, i) = params.first;
        max(0, i) = params.second;
    }
}

inline void owNeuralNetwork::getTargetMinMax(owTensor<float, 2>& min, owTensor<float, 2>& max) const {
    if (!m_dataset) return;
    auto indices = m_dataset->getUsedColumnIndices(true);
    min = owTensor<float, 2>(1, indices.size());
    max = owTensor<float, 2>(1, indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        auto params = m_dataset->getNormalizationParamsByColumnIndex(indices[i]);
        min(0, i) = params.first;
        max(0, i) = params.second;
    }
}

inline std::shared_ptr<owOptimizer> owNeuralNetwork::getOptimizer() { 
    if (!m_optimizer) m_optimizer = std::make_shared<owLBFGSOptimizer>();
    return m_optimizer; 
}

inline void owNeuralNetwork::setOptimizer(std::shared_ptr<owOptimizer> opt) { 
    m_optimizer = opt; 
    for (auto& l : m_layers) l->setOptimizer(m_optimizer.get()); 
}

inline std::vector<std::shared_ptr<owLayer>> owNeuralNetwork::getLayers() { return m_layers; }
inline void owNeuralNetwork::setLoss(std::shared_ptr<owLoss> loss) { m_loss = loss; }

inline owTensor<float, 2> owNeuralNetwork::forward(const owTensor<float, 2>& input) {
    owTensor<float, 2> currentOutput = input;
    for (auto& layer : m_layers) currentOutput = layer->forward(currentOutput);
    return currentOutput;
}

/**
 * @brief Predicts the output for a given input tensor, handling normalization and state management.
 * 
 * This function is the high-level entry point for inference. It performs the following steps:
 * 1. (Optional) Resets the network state if autoReset is true.
 * 2. Automatically normalizes the input if the associated dataset is normalized.
 * 3. Performs a forward pass through the network layers.
 * 4. Automatically inverse-normalizes the output if the associated dataset is normalized.
 * 
 * @param input The input tensor (2D: [batch_size, features]).
 * @param autoReset If true, calls reset() before prediction. 
 *                  - Set to TRUE for independent samples (e.g., sliding window forecasting where each 
 *                    window contains its own history).
 *                  - Set to FALSE for sequential/recursive forecasting (e.g., feeding the model's 
 *                    own predictions back to it) where internal state (like LSTM hidden states or 
 *                    SlidingWindow buffers) must be preserved across calls.
 * @return owTensor<float, 2> The prediction results in real-world scale (if normalized).
 */
inline owTensor<float, 2> owNeuralNetwork::predict(const owTensor<float, 2>& input, bool autoReset) {
    if (autoReset) reset();

    owTensor<float, 2> processedInput = input;
    
    // 1. Auto-Normalize input if dataset-level normalization was used
    if (m_dataset && m_dataset->isNormalized()) {
        m_dataset->normalize(processedInput);
    }

    // 2. Forward pass
    auto output = forward(processedInput);

    // 3. Auto-Inverse output if dataset-level normalization was used
    if (m_dataset && m_dataset->isNormalized()) {
        m_dataset->inverseNormalize(output);
    }
    
    return output;
}

inline owTensor<float, 2> owNeuralNetwork::predict() {
    if (!m_dataset) return owTensor<float, 2>(0, 0);
    return predict(m_dataset->getLastSample());
}

inline void owNeuralNetwork::backward(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) {
    owTensor<float, 2> grad = m_loss->gradient(prediction, target);
    float gradNormSq = 0;
    for (size_t i = 0; i < grad.size(); ++i) gradNormSq += grad.data()[i] * grad.data()[i];
    float gradNorm = std::sqrt(gradNormSq);
    if (gradNorm > 1000.0f) {
        float scale = 1000.0f / gradNorm;
        for (size_t i = 0; i < grad.size(); ++i) grad.data()[i] *= scale;
    }
    for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it) {
        if (!(*it)->isFrozen()) grad = (*it)->backward(grad);
    }
}

inline void owNeuralNetwork::trainStep() { for (auto& layer : m_layers) layer->train(); }
inline void owNeuralNetwork::reset() { for (auto& layer : m_layers) layer->reset(); }

inline size_t owNeuralNetwork::getTotalParameterCount() const {
    size_t total = 0;
    for (const auto& layer : m_layers) total += layer->getParamsCount();
    return total;
}

inline void owNeuralNetwork::getGlobalParameters(owTensor<float, 1>& target) const {
    size_t offset = 0;
    for (const auto& layer : m_layers) {
        size_t count = layer->getParamsCount();
        if (count > 0) {
            std::copy(layer->getParamsPtr(), layer->getParamsPtr() + count, target.data() + offset);
            offset += count;
        }
    }
}

inline void owNeuralNetwork::setGlobalParameters(const owTensor<float, 1>& source) {
    size_t offset = 0;
    for (auto& layer : m_layers) {
        size_t count = layer->getParamsCount();
        if (count > 0) {
            std::copy(source.data() + offset, source.data() + offset + count, layer->getParamsPtr());
            offset += count;
            layer->synchronize();
        }
    }
}

inline void owNeuralNetwork::getGlobalGradients(owTensor<float, 1>& target) const {
    size_t offset = 0;
    for (const auto& layer : m_layers) {
        size_t count = layer->getParamsCount();
        if (count > 0) {
            std::copy(layer->getGradsPtr(), layer->getGradsPtr() + count, target.data() + offset);
            offset += count;
        }
    }
}

inline float owNeuralNetwork::calculateLoss(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) {
    return m_loss ? m_loss->compute(prediction, target) : 0.0f;
}

inline const owTensor<float, 2>& owNeuralNetwork::getActiveTarget(const owTensor<float, 2>& defaultTarget) const {
    for (const auto& layer : m_layers) {
        if (layer->isFull()) return layer->getActiveTarget();
    }
    return defaultTarget;
}

inline owTensor<std::string, 1> owNeuralNetwork::getLayerNames() const {
    owTensor<std::string, 1> res(m_layers.size());
    for (size_t i = 0; i < m_layers.size(); ++i) res(i) = m_layers[i]->getLayerName();
    return res;
}

inline owTensor<float, 1> owNeuralNetwork::getNeuronNums() const {
    owTensor<float, 1> res(m_layers.size());
    for (size_t i = 0; i < m_layers.size(); ++i) res(i) = (float)m_layers[i]->getNeuronNum();
    return res;
}

inline void owNeuralNetwork::partialFit(const owTensor<float, 2>& input, const owTensor<float, 2>& target, int steps) {
    for (int i = 0; i < steps; ++i) {
        auto pred = forward(input);
        backward(pred, target);
        trainStep();
    }
    m_isPartiallyFitted = true;
}

inline bool owNeuralNetwork::saveToXML(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return false;
    file << "<owNeuralNetwork>\n";
    file << "  <Optimizer type=\"" << (m_optimizer ? m_optimizer->getOptimizerName() : "ADAM") 
         << "\" LR=\"" << (m_optimizer ? m_optimizer->getLearningRate() : 0.01f) << "\" />\n";
    file << "  <Loss type=\"" << (m_loss ? m_loss->getLossName() : "Mean Squared Error Loss") << "\" />\n";
    file << "  <Hyperparameters>\n";
    file << "    <ProjectType>" << static_cast<int>(m_projectType) << "</ProjectType>\n";
    file << "    <MaxEpochs>" << m_maxEpochs << "</MaxEpochs>\n";
    file << "    <MaxTime>" << m_maxTime << "</MaxTime>\n";
    file << "    <MinError>" << m_minError << "</MinError>\n";
    file << "    <LossStagnationTolerance>" << m_lossStagnationTolerance << "</LossStagnationTolerance>\n";
    file << "    <LossStagnationPatience>" << m_lossStagnationPatience << "</LossStagnationPatience>\n";
    file << "    <RegType>" << m_regType << "</RegType>\n";
    file << "    <PrintInterval>" << m_printInterval << "</PrintInterval>\n";
    file << "  </Hyperparameters>\n";
    file << "  <Layers count=\"" << m_layers.size() << "\">\n";
    for (const auto& layer : m_layers) {
        file << "    <Layer type=\"" << layer->getLayerName() << "\">\n";
        file << layer->toXML();
        file << "    </Layer>\n";
    }
    file << "  </Layers>\n";
    file << "</owNeuralNetwork>\n";
    file.close();
    m_isPartiallyFitted = false;
    return true;
}

inline std::shared_ptr<owOptimizer> createOptimizerByName(const std::string& name) {
    if (name == "ADAM") return std::make_shared<owADAMOptimizer>();
    if (name == "SGD") return std::make_shared<owSGDOptimizer>();
    if (name == "RMSProp") return std::make_shared<owRMSPropOptimizer>();
    if (name == "Momentum") return std::make_shared<owMomentumOptimizer>();
    if (name == "L-BFGS") return std::make_shared<owLBFGSOptimizer>();
    if (name == "Conjugate Gradient") return std::make_shared<owConjugateGradientOptimizer>();
    return std::make_shared<owADAMOptimizer>();
}

inline std::shared_ptr<owLoss> createLossByName(const std::string& name) {
    if (name == "Mean Squared Error Loss") return std::make_shared<owMeanSquaredErrorLoss>();
    if (name == "Mean Absolute Error Loss") return std::make_shared<owMeanAbsoluteErrorLoss>();
    if (name == "Huber Loss") return std::make_shared<owHuberLoss>();
    if (name == "Binary Cross-Entropy Loss") return std::make_shared<owBinaryCrossEntropyLoss>();
    if (name == "Categorical Cross-Entropy Loss") return std::make_shared<owCategoricalCrossEntropyLoss>();
    if (name == "Pinball Loss") return std::make_shared<owPinballLoss>();
    if (name == "Weighted Mean Squared Error Loss") return std::make_shared<owWeightedMeanSquaredErrorLoss>();
    if (name == "Margin Ranking Loss") return std::make_shared<owMarginRankingLoss>();
    return std::make_shared<owMeanSquaredErrorLoss>();
}

inline std::shared_ptr<owLayer> createLayerByName(const std::string& type, size_t inputSize = 0) {
    if (type == "Linear Layer") return std::make_shared<owLinearLayer>(inputSize, 1);
    if (type == "Normalization Layer") return std::make_shared<owNormalizationLayer>(inputSize);
    if (type == "Inverse Normalization Layer") return std::make_shared<owInverseNormalizationLayer>(inputSize);
    if (type == "Probability Layer") return std::make_shared<owProbabilityLayer>();
    if (type == "LSTM Layer") return std::make_shared<owLSTMLayer>(inputSize, 1);
    if (type == "Smoothing Layer") return std::make_shared<owSmoothingLayer>();
    if (type == "Rescaling Layer") return std::make_shared<owRescalingLayer>(1.0f, 0.0f);
    if (type == "Ranking Layer") return std::make_shared<owRankingLayer>(inputSize > 0 ? inputSize : 1);
    if (type == "Quantile Layer") return std::make_shared<owQuantileLayer>();
    if (type == "Principal Component Analysis Layer") return std::make_shared<owPrincipalComponentAnalysisLayer>(inputSize > 0 ? inputSize : 1, 1);
    if (type == "Projection Layer") return std::make_shared<owProjectionLayer>(inputSize > 0 ? inputSize : 1, 1);
    if (type == "Distance Layer") return std::make_shared<owDistanceLayer>(inputSize > 0 ? inputSize : 1, 1);
    if (type == "Position Encoding Layer") return std::make_shared<owPositionEncodingLayer>(1, 1);
    if (type == "Multi-Head Attention Layer") return std::make_shared<owMultiHeadAttentionLayer>(1, 1);
    if (type == "DateTime Encoding Layer") return std::make_shared<owDateTimeEncodingLayer>();
    if (type == "Cluster Layer") return std::make_shared<owClusterLayer>(inputSize > 0 ? inputSize : 1, 1);
    if (type == "Clipping Layer") return std::make_shared<owClippingLayer>(0.0f, 1.0f);
    if (type == "Bounding Layer") return std::make_shared<owBoundingLayer>(0.0f, 1.0f);
    if (type == "Attention Layer") return std::make_shared<owAttentionLayer>(inputSize > 0 ? inputSize : 1);
    if (type == "Sliding Window Layer") return std::make_shared<owSlidingWindowLayer>();
    if (type == "Sliding Window View Layer") return std::make_shared<owSlidingWindowViewLayer>();
    if (type == "Cache Layer") return std::make_shared<owCacheLayer>();
    if (type == "Trend Layer") return std::make_shared<owTrendLayer>();
    if (type == "Anomaly Detection Layer") return std::make_shared<owAnomalyDetectionLayer>();
    if (type == "Affine Layer") return std::make_shared<owAffineLayer>();
    if (type == "Addition Layer") return std::make_shared<owAdditionLayer>(inputSize > 0 ? inputSize : 1);
    if (type == "Concatenate Layer") return std::make_shared<owConcatenateLayer>();
    return nullptr;
}

inline bool owNeuralNetwork::loadFromXML(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string xml = buffer.str();
    std::function<std::shared_ptr<owLayer>(const std::string&, const std::string&)> parseLayer;
    parseLayer = [&](const std::string& layerTag, const std::string& layerContent) -> std::shared_ptr<owLayer> {
        std::string type = owLayer::getAttr(layerTag, "type");
        auto layer = createLayerByName(type, 1);
        if (!layer) return nullptr;
        if (type == "Concatenate Layer") {
            auto concat = std::static_pointer_cast<owConcatenateLayer>(layer);
            std::vector<std::shared_ptr<owConcatenateLayer::owBranch>> branches;
            std::string countStr = owLayer::getNestedTagContent(layerContent, "BranchCount");
            size_t branchCount = countStr.empty() ? 0 : std::stoul(countStr);
            for (size_t i = 0; i < branchCount; ++i) {
                std::string branchTag = "Branch_" + std::to_string(i);
                size_t bStart = layerContent.find("<" + branchTag);
                if (bStart != std::string::npos) {
                    size_t bEndTag = layerContent.find(">", bStart);
                    std::string subTag = layerContent.substr(bStart, bEndTag - bStart + 1);
                    std::string subContent = owLayer::getNestedTagContent(layerContent, branchTag);
                    auto b = std::make_shared<owConcatenateLayer::owBranch>();
                    b->fromXML(subContent);
                    std::string enabledStr = owLayer::getAttr(subTag, "enabled");
                    if (!enabledStr.empty()) b->setEnabled(std::stoi(enabledStr) == 1);
                    branches.push_back(b);
                }
            }
            concat->setBranches(branches);
        }
        layer->fromXML(layerContent);
        return layer;
    };
    size_t optPos = xml.find("<Optimizer");
    if (optPos != std::string::npos) {
        std::string optLine = xml.substr(optPos, xml.find("/>", optPos) - optPos);
        std::string type = owLayer::getAttr(optLine, "type");
        float lr = std::stof(owLayer::getAttr(optLine, "LR"));
        if (!type.empty()) setOptimizer(createOptimizerByName(type));
        if (m_optimizer) m_optimizer->setLearningRate(lr);
    }
    size_t lossPos = xml.find("<Loss");
    if (lossPos != std::string::npos) {
        std::string lossLine = xml.substr(lossPos, xml.find("/>", lossPos) - lossPos);
        std::string type = owLayer::getAttr(lossLine, "type");
        if (!type.empty()) setLoss(createLossByName(type));
    }
    std::string hpContent = owLayer::getTagContent(xml, "Hyperparameters");
    if (!hpContent.empty()) {
        std::string val;
        val = owLayer::getTagContent(hpContent, "ProjectType"); if (!val.empty()) m_projectType = static_cast<owProjectType>(std::stoi(val));
        val = owLayer::getTagContent(hpContent, "MaxEpochs"); if (!val.empty()) m_maxEpochs = std::stoi(val);
        val = owLayer::getTagContent(hpContent, "MaxTime"); if (!val.empty()) m_maxTime = std::stod(val);
        val = owLayer::getTagContent(hpContent, "MinError"); if (!val.empty()) m_minError = std::stof(val);
        val = owLayer::getTagContent(hpContent, "LossStagnationTolerance"); if (!val.empty()) m_lossStagnationTolerance = std::stof(val);
        val = owLayer::getTagContent(hpContent, "LossStagnationPatience"); if (!val.empty()) m_lossStagnationPatience = std::stoi(val);
        val = owLayer::getTagContent(hpContent, "RegType"); if (!val.empty()) m_regType = std::stoi(val);
        val = owLayer::getTagContent(hpContent, "PrintInterval"); if (!val.empty()) m_printInterval = std::stoi(val);
    }
    m_layers.clear();
    std::string layersContent = owLayer::getTagContent(xml, "Layers");
    size_t pos = 0;
    while ((pos = layersContent.find("<Layer", pos)) != std::string::npos) {
        size_t lineEnd = layersContent.find(">", pos);
        std::string layerTag = layersContent.substr(pos, lineEnd - pos + 1);
        size_t searchPos = lineEnd + 1;
        int depth = 1;
        size_t layerEnd = std::string::npos;
        while (depth > 0) {
            size_t nextOpen = layersContent.find("<Layer", searchPos);
            size_t nextClose = layersContent.find("</Layer>", searchPos);
            if (nextClose == std::string::npos) break;
            if (nextOpen != std::string::npos && nextOpen < nextClose) {
                depth++; searchPos = nextOpen + 6;
            } else {
                depth--;
                if (depth == 0) layerEnd = nextClose;
                searchPos = nextClose + 8;
            }
        }
        if (layerEnd == std::string::npos) break;
        std::string layerContent = layersContent.substr(lineEnd + 1, layerEnd - (lineEnd + 1));
        auto layer = parseLayer(layerTag, layerContent);
        if (layer) addLayer(layer);
        pos = layerEnd + 8;
    }
    return true;
}

inline void owNeuralNetwork::createNeuralNetwork(const std::vector<int>& hiddenSizes, 
                                               const std::string& hiddenAct, 
                                               const std::string& outputAct,
                                               bool /*useNormalization*/) {
    m_projectType = owProjectType::CUSTOM;
    m_layers.clear();
    int inputSize = m_dataset->getInputVariableNum();
    int targetSize = m_dataset->getTargetVariableNum();

    int currentIn = inputSize;
    for (int hSize : hiddenSizes) {
        auto layer = std::make_shared<owLinearLayer>(currentIn, hSize);
        layer->initializeWeightsWithRNG(m_rng); 
        layer->setActivationByName(hiddenAct);
        addLayer(layer);
        currentIn = hSize;
    }

    auto outLayer = std::make_shared<owLinearLayer>(currentIn, targetSize);
    outLayer->initializeWeightsWithRNG(m_rng);
    outLayer->setActivationByName(outputAct);
    addLayer(outLayer);
}

inline void owNeuralNetwork::createNeuralNetwork(owProjectType type, const std::vector<int>& hiddenSizes, int /*windowSize*/) {
    m_projectType = type; m_layers.clear();
    int inputSize = m_dataset->getInputVariableNum(), targetSize = m_dataset->getTargetVariableNum();
    if (type == owProjectType::CLUSTERING) {
        int latentDim = hiddenSizes.empty() ? inputSize : hiddenSizes[0], numClusters = targetSize;
        addLayer(std::make_shared<owProjectionLayer>(inputSize, latentDim));
        addLayer(std::make_shared<owClusterLayer>(latentDim, numClusters));
        addLayer(std::make_shared<owDistanceLayer>(numClusters, numClusters));
        return;
    }
    if (type == owProjectType::ANOMALY_DETECTION) {
        int latentDim = hiddenSizes.empty() ? inputSize : hiddenSizes[0];
        addLayer(std::make_shared<owProjectionLayer>(inputSize, latentDim));
        addLayer(std::make_shared<owAnomalyDetectionLayer>());
        return;
    }
    int currentIn = inputSize;
    for (int hSize : hiddenSizes) {
        auto layer = std::make_shared<owLinearLayer>(currentIn, hSize);
        layer->initializeWeightsWithRNG(m_rng); layer->setActivationByName("ReLU");
        addLayer(layer); currentIn = hSize;
    }
    auto outLayer = std::make_shared<owLinearLayer>(currentIn, targetSize);
    outLayer->initializeWeightsWithRNG(m_rng);
    if (type == owProjectType::CLASSIFICATION) {
        outLayer->setActivationByName("Sigmoid");
        if (targetSize > 1) addLayer(std::make_shared<owProbabilityLayer>());
    } else outLayer->setActivationByName("Identity");
    addLayer(outLayer);
}

inline void owNeuralNetwork::train() {
    if (!m_dataset || !m_optimizer || !m_loss) return;
    auto startTime = std::chrono::high_resolution_clock::now();
    setTraining(true); 
    bool hasCache = false;
    for (auto& l : m_layers) if (l->getLayerName() == "Cache Layer") { hasCache = true; break; }
    if (hasCache) {
        auto fullData = m_dataset->getData();
        std::vector<int> inputIndices = m_dataset->getUsedColumnIndices(false);
        std::vector<int> targetIndices = m_dataset->getUsedColumnIndices(true);
        if (inputIndices.empty()) inputIndices = targetIndices;
        if (inputIndices.empty()) return;
        owTensor<float, 2> fullIn(fullData.shape()[0], inputIndices.size());
        owTensor<float, 2> fullTarget(fullData.shape()[0], targetIndices.size());
        for (size_t i = 0; i < fullData.shape()[0]; ++i) {
            for (size_t j = 0; j < inputIndices.size(); ++j) fullIn(i, j) = fullData(i, (size_t)inputIndices[j]);
            for (size_t j = 0; j < targetIndices.size(); ++j) fullTarget(i, j) = fullData(i, (size_t)targetIndices[j]);
        }
        reset(); for (auto& l : m_layers) l->setTarget(&fullTarget);
        forward(fullIn); for (auto& l : m_layers) l->lockCache(); reset(); 
    }
    if (m_optimizer->supportsGlobalOptimization()) m_optimizer->optimizeGlobal(this, m_dataset.get());
    else runStandardTrainingLoop();
    setTraining(false); 
    for (auto& layer : m_layers) layer->setPlaybackMode(false);
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    m_actualTrainingTime = elapsed.count();
    if (m_enablePrinting) {
        auto trainReport = evaluatePerformance(m_dataset->getTrainInput(), m_dataset->getTrainTarget());
        float trainMAPE = (trainReport.realScaleMape > 0.0f) ? trainReport.realScaleMape : trainReport.mape;
        float vMAPE = 0.0f; auto vIn = m_dataset->getValInput();
        if (vIn.shape()[0] > 0) {
            auto valReport = evaluatePerformance(vIn, m_dataset->getValTarget());
            vMAPE = (valReport.realScaleMape > 0.0f) ? valReport.realScaleMape : valReport.mape;
        }
        std::cout << "\n--- Training Summary ---" << std::endl;
        std::cout << "Finish Reason: " << m_finishReason << std::endl;
        std::cout << "Avg. Train Loss: " << std::fixed << std::setprecision(2) << trainMAPE << "% (MAPE)" << std::endl;
        if (vIn.shape()[0] > 0) std::cout << "Avg. Val Loss: " << std::fixed << std::setprecision(2) << vMAPE << "% (MAPE)" << std::endl;
        std::cout << "Total Time: ";
        if (m_actualTrainingTime < 60.0) std::cout << std::fixed << std::setprecision(2) << m_actualTrainingTime << "s" << std::endl;
        else if (m_actualTrainingTime < 3600.0) {
            int m = static_cast<int>(m_actualTrainingTime) / 60; double s = std::fmod(m_actualTrainingTime, 60.0);
            std::cout << m << "m " << std::fixed << std::setprecision(2) << s << "s" << std::endl;
        } else {
            int h = static_cast<int>(m_actualTrainingTime) / 3600, m = (static_cast<int>(m_actualTrainingTime) % 3600) / 60;
            double s = std::fmod(m_actualTrainingTime, 60.0);
            std::cout << h << "h " << m << "m " << std::fixed << std::setprecision(2) << s << "s" << std::endl;
        }
        std::cout << "Total Epochs: " << m_actualEpochs << std::endl;
        if (m_actualEpochs > 0) std::cout << "Avg Time/Epoch: " << std::fixed << std::setprecision(2) << (m_actualTrainingTime / m_actualEpochs) * 1000.0 << "ms" << std::endl;
        std::cout << "------------------------\n" << std::endl;
    }
}

inline void owNeuralNetwork::runStandardTrainingLoop() {
    auto trainIn = m_dataset->getTrainInput(), trainTarget = m_dataset->getTrainTarget();
    auto startTime = std::chrono::high_resolution_clock::now();
    float bestLoss = std::numeric_limits<float>::max(); int patienceCounter = 0;
    for (int epoch = 1; epoch <= m_maxEpochs; ++epoch) {
        setTraining(true); reset(); 
        float loss = 0.0f, valLoss = 0.0f;
        for (auto& layer : m_layers) layer->setTarget(&trainTarget);
        const owTensor<float, 2>* activeTarget = &trainTarget;
        std::shared_ptr<owCacheLayer> activeCache = nullptr;
        for (auto& layer : m_layers) {
            auto cache = std::dynamic_pointer_cast<owCacheLayer>(layer);
            if (cache && cache->isFull()) activeCache = cache;
        }
        auto pred = forward(trainIn);
        if (activeCache) activeTarget = &activeCache->getActiveTarget();
        loss = calculateLoss(pred, *activeTarget); backward(pred, *activeTarget);
        
        if (m_enablePrinting && (epoch == 1 || epoch % m_printInterval == 0)) {
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> currentElapsed = now - startTime;
            printTrainingStatus(epoch, loss, valLoss, currentElapsed.count());
        }

        if (m_minMape > 0.0f) {
            float currentMape = 0.0f; size_t n = pred.shape()[0], outDim = pred.shape()[1];
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j < outDim; ++j) {
                    float p = pred(i, j), t = (*activeTarget)(i, j);
                    if (std::abs(t) > 1e-7f) currentMape += std::abs((p - t) / t);
                }
            }
            currentMape = (currentMape / (n * outDim)) * 100.0f;
            if (currentMape <= m_minMape) {
                m_finishReason = "Minimum Error"; m_actualEpochs = epoch;
                if (m_enablePrinting && epoch % m_printInterval != 0) {
                    auto now = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> currentElapsed = now - startTime;
                    printTrainingStatus(epoch, loss, valLoss, currentElapsed.count());
                }
                break;
            }
        }
        if (epoch == 1) { for (auto& layer : m_layers) layer->lockCache(); }
        for (auto& layer : m_layers) {
            auto concat = std::dynamic_pointer_cast<owConcatenateLayer>(layer);
            if (concat) {
                for (auto& branch : concat->getBranches()) {
                    if (branch->isIndependentExpertMode() && branch->getConvergenceThreshold() > 0 && !branch->isFrozen()) {
                        float localErr = branch->computeLocalLoss(trainTarget);
                        if (localErr < branch->getConvergenceThreshold()) branch->setFrozen(true);
                    }
                }
            }
            if (layer->isIndependentExpertMode() && layer->getConvergenceThreshold() > 0 && !layer->isFrozen()) {
                float localErr = 1e30f; auto seq = std::dynamic_pointer_cast<owSequentialLayer>(layer);
                if (seq) localErr = seq->computeLocalLoss(trainTarget);
                if (localErr < layer->getConvergenceThreshold()) layer->setFrozen(true);
            }
        }
        trainStep(); m_lastTrainLoss = loss;
        auto valIn = m_dataset->getValInput();
        if (valIn.size() > 0) {
            setTraining(false); auto valTarget = m_dataset->getValTarget(); auto valPred = forward(valIn);
            valLoss = calculateLoss(valPred, valTarget); m_lastValLoss = valLoss; setTraining(true);
        }
        if (m_lossStagnationEnabled) {
            if (loss < bestLoss - m_lossStagnationTolerance) { bestLoss = loss; patienceCounter = 0; }
            else patienceCounter++;
            if (patienceCounter >= m_lossStagnationPatience) {
                m_finishReason = "Loss Stagnation"; m_actualEpochs = epoch;
                if (m_enablePrinting && epoch % m_printInterval != 0) {
                    auto now = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> currentElapsed = now - startTime;
                    printTrainingStatus(epoch, loss, valLoss, currentElapsed.count());
                }
                break;
            }
        }
        if (m_minError > 0 && m_lastTrainLoss <= m_minError) {
            m_finishReason = "Minimum Error"; m_actualEpochs = epoch;
            if (m_enablePrinting && epoch % m_printInterval != 0) {
                auto now = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> currentElapsed = now - startTime;
                printTrainingStatus(epoch, loss, valLoss, currentElapsed.count());
            }
            break;
        }
        m_actualEpochs = epoch; m_finishReason = "Maximum Epoch Num";
        if (epoch == m_maxEpochs && m_enablePrinting && epoch % m_printInterval != 0) {
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> currentElapsed = now - startTime;
            printTrainingStatus(epoch, loss, valLoss, currentElapsed.count());
        }
    }
}

inline EvaluationReport owNeuralNetwork::evaluatePerformance(const owTensor<float, 2>& input, const owTensor<float, 2>& target, float tolerance) {
    EvaluationReport report; setTraining(false); reset(); 
    auto pred = forward(input); size_t n = input.shape()[0], outDim = target.shape()[1];
    float mse = 0, mape = 0; int correct = 0;
    float realMape = 0.0f; bool canInverse = m_dataset && m_dataset->isNormalized();
    auto realPred = pred; auto realTarget = target;
    if (canInverse) { m_dataset->inverseNormalize(realPred); m_dataset->inverseNormalize(realTarget); }
    for (size_t i = 0; i < n; ++i) {
        bool rowCorrect = true;
        for (size_t j = 0; j < outDim; ++j) {
            float p = pred(i, j), t = target(i, j), diff = p - t;
            mse += diff * diff; if (std::abs(t) > 1e-7f) mape += std::abs(diff / t);
            if (canInverse) {
                float rp = realPred(i, j), rt = realTarget(i, j);
                if (std::abs(rt) > 1e-7f) realMape += std::abs((rp - rt) / rt);
            }
            float threshold = (std::abs(t) < 1e-7f) ? tolerance : std::abs(t * tolerance);
            if (std::abs(diff) > threshold) rowCorrect = false;
        }
        if (rowCorrect) correct++;
    }
    report.rmse = std::sqrt(mse / (n * outDim));
    report.mape = (mape / (n * outDim)) * 100.0f;
    report.realScaleMape = (realMape / (n * outDim)) * 100.0f;
    report.accuracy = (float)correct / n;
    return report;
}

inline EvaluationReport owNeuralNetwork::evaluatePerformance(float tolerance) {
    if (!m_dataset) return EvaluationReport();
    auto testIn = m_dataset->getTestInput(), testOut = m_dataset->getTestTarget();
    return evaluatePerformance(testIn, testOut, tolerance);
}

inline std::string owNeuralNetwork::predictLabel(const owTensor<float, 2>& input, int targetVarIdx) {
    auto pred = predict(input); if (!m_dataset) return "";
    int actualColIdx = m_dataset->getTargetColumnIndex(targetVarIdx);
    return m_dataset->getLabelName(actualColIdx, pred(0, targetVarIdx));
}

inline owTensor<float, 2> owNeuralNetwork::forecast(int steps, bool unscale) {
    if (!m_dataset) return owTensor<float, 2>(0, 0);
    return forecast(m_dataset->getLastSample(), steps, unscale);
}

inline owTensor<float, 2> owNeuralNetwork::forecast(const owTensor<float, 2>& initialSample, int steps, bool unscale) {
    if (steps <= 0 || initialSample.shape()[0] == 0 || initialSample.shape()[1] == 0) return owTensor<float, 2>(0, 0);
    reset();
    owTensor<float, 2> currentInput = initialSample;
    size_t inputFeatures = currentInput.shape()[1];

    auto firstPred = forward(currentInput);
    size_t targetSize = firstPred.shape()[1];
    if (targetSize == 0) return owTensor<float, 2>(0, 0);

    // Direct Multi-Horizon model: if targetSize > 1 and already contains enough steps
    if (targetSize >= (size_t)steps && targetSize > 1) {
        owTensor<float, 2> results(steps, 1);
        for (int i = 0; i < steps; ++i) {
            results(i, 0) = firstPred(0, i);
        }
        if (unscale && m_dataset && m_dataset->isNormalized()) {
            m_dataset->inverseNormalize(results);
        }
        return results;
    }

    // Autoregressive Roll-Forward (Recursive multi-step forecast)
    owTensor<float, 2> results(steps, targetSize);
    for (int i = 0; i < steps; ++i) {
        auto pred = forward(currentInput);
        for (size_t j = 0; j < targetSize; ++j) {
            results(i, j) = pred(0, j);
        }

        // Sliding window shift for next horizon step
        if (targetSize == inputFeatures) {
            currentInput = pred;
        } else if (targetSize < inputFeatures) {
            // Shift history window left by targetSize
            for (size_t k = 0; k < inputFeatures - targetSize; ++k) {
                currentInput(0, k) = currentInput(0, k + targetSize);
            }
            // Inject newest prediction into the most recent slots
            for (size_t j = 0; j < targetSize; ++j) {
                currentInput(0, inputFeatures - targetSize + j) = pred(0, j);
            }
        } else {
            for (size_t j = 0; j < inputFeatures; ++j) {
                currentInput(0, j) = pred(0, j);
            }
        }
    }

    if (unscale && m_dataset && m_dataset->isNormalized()) {
        m_dataset->inverseNormalize(results);
    }
    return results;
}

inline void owNeuralNetwork::printEvaluationReport(const EvaluationReport& report) const {
    std::cout << "--- Evaluation Report ---" << std::endl;
    std::cout << "RMSE: " << report.rmse << std::endl;
    std::cout << "MAPE: " << report.mape << "%" << std::endl;
    std::cout << "Accuracy (within tolerance): " << report.accuracy * 100.0f << "%" << std::endl;
}

inline std::shared_ptr<owActivation> owNeuralNetwork::createActivationByName(const std::string& name) {
    if (name == "ReLU") return std::make_shared<owReLUActivation>();
    if (name == "Sigmoid") return std::make_shared<owSigmoidActivation>();
    if (name == "Tanh") return std::make_shared<owTanhActivation>();
    if (name == "LeakyReLU") return std::make_shared<owLeakyReLUActivation>();
    return std::make_shared<owIdentityActivation>();
}

inline void owNeuralNetwork::printTrainingStatus(int epoch, float trainLoss, float valLoss, double elapsedTime) {
    float scale = 1.0f;
    if (m_dataset && !m_dataset->isNormalized()) {
        auto params = m_dataset->getNormalizationParamsByColumnIndex(m_dataset->getTargetColumnIndex(0));
        float range = params.second - params.first;
        if (range > 0) scale = range * range;
    }

    std::cout << "Epoch " << epoch << " | Train Loss: " << (trainLoss / scale);
    if (valLoss > 0.0f) std::cout << " | Val Loss: " << (valLoss / scale);
    std::cout << " | Total Time: " << elapsedTime << "s" << std::endl;
}

}
