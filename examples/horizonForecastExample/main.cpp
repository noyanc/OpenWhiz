/*
 * horizonForecastExample
 * 
 * Demonstrates OpenWhiz future forecasting capabilities:
 * 1. Autoregressive Roll-Forward Multi-Step Forecasting (nn.forecast(steps))
 * 2. Direct Multi-Horizon Forecasting (prepareForecastData(windowSize, horizon))
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <memory>
#include "OpenWhiz/openwhiz.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Generates realistic daily energy demand data (trend + cyclical seasonality + noise)
void generateEnergyDemandCSV(const std::string& filename, int days) {
    std::ofstream file(filename);
    file << "Day,Demand\n";
    for (int t = 0; t < days; ++t) {
        float trend = 100.0f + 0.05f * t; // subtle upward trend
        float weekly = 20.0f * std::sin(2.0f * (float)M_PI * t / 7.0f); // 7-day cycle
        float seasonal = 15.0f * std::cos(2.0f * (float)M_PI * t / 30.0f); // 30-day cycle
        float noise = 2.0f * std::sin((float)(t * 13 % 17));
        float demand = trend + weekly + seasonal + noise;
        file << (t + 1) << "," << std::fixed << std::setprecision(2) << demand << "\n";
    }
    file.close();
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "   OpenWhiz Advanced Horizon & Future Forecasting Demo    " << std::endl;
    std::cout << "==========================================================\n" << std::endl;

    const std::string csvPath = "energy_demand.csv";
    const int totalDays = 350;
    generateEnergyDemandCSV(csvPath, totalDays);
    std::cout << "[Data] Generated " << totalDays << " days of synthetic energy demand data.\n" << std::endl;

    // =========================================================================
    // PART 1: Autoregressive Roll-Forward Forecasting (nn.forecast(steps))
    // =========================================================================
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << " PART 1: Autoregressive Roll-Forward Horizon Forecast     " << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << "Strategy: Model learns 1-step prediction (W=7 -> 1)." << std::endl;
    std::cout << "At inference, the sliding window shifts forward automatically." << std::endl;

    auto dataset1 = std::make_shared<ow::owDataset>();
    dataset1->setDelimiter(',');
    if (!dataset1->loadFromCSV(csvPath, /*hasHeader=*/true, /*autoNormalize=*/true)) {
        std::cerr << "Failed to load CSV." << std::endl;
        return 1;
    }

    // Exclude Day index, predict Demand
    dataset1->setColumnUsage("Day", ow::ColumnUsage::UNUSED);
    dataset1->setTargetVariableNum(1);

    const int windowSize1 = 7; // 7-day history lookback
    dataset1->prepareForecastData(windowSize1, /*horizon=*/1);
    dataset1->setRatios(0.80f, 0.10f, 0.10f, false);

    std::cout << "Dataset configured: Input Features (Lags) = " << dataset1->getInputVariableNum()
              << ", Target Variables = " << dataset1->getTargetVariableNum() << std::endl;

    // Build model
    ow::owNeuralNetwork nn1;
    nn1.setDataset(dataset1);
    nn1.setProjectType(ow::owProjectType::FORECASTING);
    nn1.createNeuralNetwork(ow::owProjectType::FORECASTING, {32, 16});
    nn1.setOptimizer(std::make_shared<ow::owLBFGSOptimizer>(1.0f, 30));
    nn1.setMaximumEpochNum(150);
    nn1.setPrintEpochInterval(50);

    std::cout << "Training model for 1-step forecasting..." << std::endl;
    nn1.train();

    // Perform 10-day future horizon projection
    const int forecastSteps = 10;
    std::cout << "\nExecuting nn.forecast(" << forecastSteps << ") into the future..." << std::endl;
    ow::owTensor<float, 2> forecast10Days = nn1.forecast(forecastSteps, /*unscale=*/true);

    std::cout << "\n--- Projected Future 10 Days (Physical Units: kWh) ---" << std::endl;
    for (int s = 0; s < forecastSteps; ++s) {
        std::cout << "  Day t+" << std::setw(2) << (s + 1)
                  << " Forecast: " << std::fixed << std::setprecision(2)
                  << forecast10Days(s, 0) << " kWh" << std::endl;
    }

    // =========================================================================
    // PART 2: Direct Multi-Horizon Forecasting (W=7 -> H=5)
    // =========================================================================
    std::cout << "\n----------------------------------------------------------" << std::endl;
    std::cout << " PART 2: Direct Multi-Horizon Forecasting (Direct Multi-Output)" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << "Strategy: Model directly predicts 5 future days in a single forward pass." << std::endl;

    auto dataset2 = std::make_shared<ow::owDataset>();
    dataset2->setDelimiter(',');
    dataset2->loadFromCSV(csvPath, true, true);
    dataset2->setColumnUsage("Day", ow::ColumnUsage::UNUSED);

    const int windowSize2 = 7;
    const int horizon2 = 5;
    dataset2->prepareForecastData(windowSize2, horizon2);
    dataset2->setRatios(0.80f, 0.10f, 0.10f, false);

    std::cout << "Dataset configured: Input Features = " << dataset2->getInputVariableNum()
              << ", Direct Horizon Targets = " << dataset2->getTargetVariableNum() << std::endl;

    ow::owNeuralNetwork nn2;
    nn2.setDataset(dataset2);
    nn2.setProjectType(ow::owProjectType::FORECASTING);
    nn2.createNeuralNetwork(ow::owProjectType::FORECASTING, {32, 16});
    nn2.setOptimizer(std::make_shared<ow::owLBFGSOptimizer>(1.0f, 30));
    nn2.setMaximumEpochNum(150);
    nn2.setPrintEpochInterval(50);

    std::cout << "Training direct multi-horizon model..." << std::endl;
    nn2.train();

    std::cout << "\nExecuting direct 5-day horizon forecast..." << std::endl;
    ow::owTensor<float, 2> direct5Days = nn2.forecast(horizon2, /*unscale=*/true);

    std::cout << "\n--- Direct 5-Day Multi-Horizon Outputs ---" << std::endl;
    for (int h = 0; h < horizon2; ++h) {
        std::cout << "  Horizon +" << (h + 1) << " Target: "
                  << std::fixed << std::setprecision(2)
                  << direct5Days(h, 0) << " kWh" << std::endl;
    }

    std::cout << "\n==========================================================" << std::endl;
    std::cout << "   Horizon & Future Forecasting Demo Completed Successfully! " << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}
