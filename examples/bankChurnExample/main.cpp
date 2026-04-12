#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "OpenWhiz/openwhiz.hpp"


int main() {
    std::cout << "=== OpenWhiz Bank Churn Classification Example ===\n" << std::endl;

    const std::string csvFile = "C:/dev/OpenWhiz/examples/bankChurnExample/bankchurn2.csv";

    ow::owNeuralNetwork nn;
    
    // 1. Setup Data
    nn.getDataset()->setAutoNormalizeEnabled(true);
    if (!nn.loadData(csvFile)) {
        std::cerr << "Failed to load data!" << std::endl;
        return -1;
    }
    
    // Column configuration as requested:
    // USED (default), UNUSED, ORDERING
    // Note: customer_id is already filtered out by loadData (ends in ID)
    
    // Set the target variable num (churn is the last column)
    nn.getDataset()->setTargetVariableNum(1);

    // 2. Build Network Architecture
    nn.createNeuralNetwork(ow::owProjectType::CLASSIFICATION, {64, 32});
    
    // 3. Configure Training
    nn.setLoss(std::make_shared<ow::owBinaryCrossEntropyLoss>());
    nn.setOptimizer(std::make_shared<ow::owLBFGSOptimizer>());

    // 4. Train
    nn.train();

    std::cout << "Detected Delimiter: '" << nn.getDataset()->getDelimiter() << "'" << std::endl;

    // 5. Custom Classification Evaluation using the new 1-parameter overload
    auto eval = nn.evaluatePerformance(); 
    
    std::cout << "\nFinal Performance on Test Set:" << std::endl;
    nn.printEvaluationReport(eval);

    // 6. Manual Prediction with Label Name (Showing Predicted vs Actual)
    auto testIn = nn.getDataset()->getTestInput();
    auto testOut = nn.getDataset()->getTestTarget();
    
    if (testIn.shape()[0] > 0) {
        ow::owTensor<float, 2> sample(1, testIn.shape()[1]);
        for(size_t i=0; i<sample.shape()[1]; ++i) sample(0, i) = testIn(0, i);
        
        std::string predLabel = nn.predictLabel(sample);
        
        // Get actual label name for the first test target
        int targetColIdx = nn.getDataset()->getTargetColumnIndex(0);
        std::string actualLabel = nn.getDataset()->getLabelName(targetColIdx, testOut(0, 0));
        
        std::cout << "Manual Prediction for first test sample:" << std::endl;
        std::cout << "  - Predicted: " << predLabel << std::endl;
        std::cout << "  - Actual:    " << actualLabel << std::endl;
    }
    
    return 0;
}
