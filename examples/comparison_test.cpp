#include <iostream>
#include <iomanip>
#include "OpenWhiz/openwhiz.hpp"

int main() {
    const std::string csvFile = "C:/dev/OpenWhiz/examples/cac40ForecastExample/cac40_3years.csv";

    // 1. DATASET NORMALIZATION
    auto ds1 = std::make_shared<ow::owDataset>();
    ds1->loadFromCSV(csvFile, true, true); 
    ds1->setColumnUsage("Date", ow::ColumnUsage::UNUSED);
    ds1->setTargetVariableNum(1);

    // 2. LAYER NORMALIZATION
    auto ds2 = std::make_shared<ow::owDataset>();
    ds2->loadFromCSV(csvFile, true, false); 
    ds2->setColumnUsage("Date", ow::ColumnUsage::UNUSED);
    ds2->setTargetVariableNum(1);
    
    ow::owNeuralNetwork nn;
    nn.setDataset(ds2);
    auto normLayer = std::make_shared<ow::owNormalizationLayer>();
    nn.addLayer(normLayer); 
    
    std::cout << "Comparing normalized Price column values:" << std::endl;
    std::cout << std::setw(15) << "Dataset Method" << std::setw(15) << "Layer Method" << std::setw(15) << "Difference" << std::endl;
    
    auto data1 = ds1->getData();
    auto data2 = ds2->getData();

    for(int i = 0; i < 5; ++i) {
        float val1 = data1(i, 1); 
        
        ow::owTensor<float, 2> rowIn(1, 1);
        rowIn(0, 0) = data2(i, 1);
        auto rowOut = normLayer->forward(rowIn);
        float val2 = rowOut(0, 0);
        
        std::cout << std::fixed << std::setprecision(6) 
                  << std::setw(15) << val1 
                  << std::setw(15) << val2 
                  << std::setw(15) << std::abs(val1 - val2) << std::endl;
    }

    return 0;
}
