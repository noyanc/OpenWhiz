/*
 * owDataset.hpp
 *
 *  Created on: Dec 16, 2025
 *      Author: Noyan Culum, AITIAL
 */


#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <random>
#include <chrono>
#include "../core/owTensor.hpp"

/**
 * @file owDataset.hpp
 * @brief Data management and preprocessing utilities for OpenWhiz.
 */

namespace ow {

/**
 * @enum DataType
 * @brief Supported data types for dataset columns.
 */
enum class DataType { 
    Numeric,   ///< Continuous or discrete numerical values.
    Datetime,  ///< Date and time strings (to be encoded).
    Text       ///< Categorical or raw text data.
};

/**
 * @enum Ordering
 * @brief Categorical ordering strategies.
 */
enum class Ordering { 
    Standard,     ///< No special ordering.
    Categorical,  ///< Unordered categories (Nominal).
    Ordered       ///< Categories with a specific sequence (Ordinal).
};

/**
 * @enum ColumnUsage
 * @brief Usage status of a column for training and calculation.
 */
enum class ColumnUsage {
    USED,     ///< Column is used for training and calculation.
    UNUSED,   ///< Column is loaded but explicitly excluded from training and calculation.
    ORDERING  ///< Column is used as a sequence marker (e.g., Step No, Time), not for training.
};

/**
 * @enum SampleType
 * @brief Categorization for dataset splitting.
 */
enum class SampleType { 
    Training,    ///< Data used for model parameter updates.
    Validation,  ///< Data used for hyperparameter tuning and early stopping.
    Test         ///< Unseen data for final performance evaluation.
};

/**
 * @enum ImputationStrategy
 * @brief Strategies for handling missing data (NaN/Empty).
 */
enum class ImputationStrategy { 
    Mean,         ///< Replace with the column mean.
    Zero,         ///< Replace with 0.0.
    ForwardFill   ///< Replace with the previous valid value.
};

/**
 * @struct ColumnInfo
 * @brief Metadata for a single dataset column.
 */
struct ColumnInfo {
    std::string name;                          ///< Name of the column (from CSV header).
    DataType type;                             ///< Interpreted data type.
    Ordering ordering;                         ///< Categorical ordering type.
    ColumnUsage usage = ColumnUsage::USED;     ///< Usage status for training.
    std::map<std::string, float> category_map; ///< Mapping from category string to float ID.
    std::vector<std::string> reverse_category_map; ///< Mapping from float ID back to category string.
    float min = 0.0f;                          ///< Minimum value observed in the data.
    float max = 1.0f;                          ///< Maximum value observed in the data.
};

/**
 * @class owDataset
 * @brief Core class for data loading, preprocessing, and management.
 */
class owDataset {
public:
    owDataset() : m_targetVariableNum(1), m_autoNormalizeEnabled(false) {
        auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        m_rng.seed(static_cast<unsigned int>(seed));
    }
    ~owDataset() = default;

    /**
     * @brief Loads and parses a CSV file.
     * 
     * Filtering: Columns ending in "ID" (case-insensitive) are not loaded.
     * Categorical: Automatically detects text columns and applies label encoding.
     * Delimiter: Automatically detected if not explicitly set.
     */
    bool loadFromCSV(const std::string& filepath, bool hasHeader = true, bool autoNormalize = false) {
        std::ifstream file(filepath);
        if (!file.is_open()) return false;
        std::string line;

        // Clear previous raw data
        m_rawCSVData.clear();

        // Automatic Delimiter Detection
        if (std::getline(file, line)) {
            char candidates[] = {',', ';', '|', '\t'};
            char best_d = m_delimiter;
            int max_c = 0;
            
            for (char cnd : candidates) {
                int count = (int)std::count(line.begin(), line.end(), cnd);
                if (count > max_c) {
                    max_c = count;
                    best_d = cnd;
                }
            }
            // If no clear winner, but the line contains commas and we are at default ';', prefer ','
            if (max_c == 0 && line.find(',') != std::string::npos) best_d = ',';

            m_delimiter = best_d;
            
            file.clear();
            file.seekg(0);
        }

        std::vector<int> col_indices; // Indices of columns to keep
        
        auto isID = [](const std::string& name) {
            if (name.length() < 2) return false;
            std::string suffix = name.substr(name.length() - 2);
            for (char &c : suffix) c = (char)std::toupper((unsigned char)c);
            return suffix == "ID";
        };

        if (hasHeader && std::getline(file, line)) {
            std::stringstream ss(line);
            std::string col;
            int idx = 0;
            while (std::getline(ss, col, m_delimiter)) {
                std::string cleaned = cleanColumnName(col);
                if (!isID(cleaned)) {
                    m_columns.push_back({cleaned, DataType::Numeric, Ordering::Standard, ColumnUsage::USED, {}, {}, 0.0f, 1.0f});
                    col_indices.push_back(idx);
                }
                idx++;
            }
        }

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string val;
            std::vector<std::string> full_row;
            while (std::getline(ss, val, m_delimiter)) full_row.push_back(val);
            
            if (m_columns.empty()) {
                for (size_t i = 0; i < full_row.size(); ++i) {
                    std::string name = "col_" + std::to_string(i);
                    if (!isID(name)) {
                        m_columns.push_back({name, DataType::Numeric, Ordering::Standard, ColumnUsage::USED, {}, {}, 0.0f, 1.0f});
                        col_indices.push_back((int)i);
                    }
                }
            }

            std::vector<std::string> filtered_row;
            for (int idx : col_indices) {
                if ((size_t)idx < full_row.size()) filtered_row.push_back(full_row[idx]);
                else filtered_row.push_back("0");
            }
            m_rawCSVData.push_back(filtered_row);
        }

        if (m_rawCSVData.empty()) return false;
        size_t rows = m_rawCSVData.size();
        size_t cols = m_columns.size();

        // Detect Data Types and Categorical Mapping
        for (size_t c = 0; c < cols; ++c) {
            bool all_numeric = true;
            for (size_t r = 0; r < rows; ++r) {
                const std::string& val = m_rawCSVData[r][c];
                if (val.empty() || val == "NaN" || val == "null" || val == "nan") continue;
                
                bool has_digit = false;
                bool dot_seen = false;
                bool valid = true;
                for (size_t i = 0; i < val.length(); ++i) {
                    if (i == 0 && (val[i] == '-' || val[i] == '+')) continue;
                    if (val[i] == '.' || val[i] == ',') {
                        if (dot_seen) { valid = false; break; }
                        dot_seen = true;
                    } else if (std::isdigit((unsigned char)val[i])) {
                        has_digit = true;
                    } else {
                        valid = false;
                        break;
                    }
                }
                if (!valid || !has_digit) { all_numeric = false; break; }
            }

            if (!all_numeric) {
                m_columns[c].type = DataType::Text;
                float next_id = 0.0f;
                for (size_t r = 0; r < rows; ++r) {
                    const std::string& val = m_rawCSVData[r][c];
                    if (m_columns[c].category_map.find(val) == m_columns[c].category_map.end()) {
                        m_columns[c].category_map[val] = next_id++;
                        m_columns[c].reverse_category_map.push_back(val);
                    }
                }
            }
        }

        m_fullData = owTensor<float, 2>(rows, cols);
        for (size_t c = 0; c < cols; ++c) {
            for (size_t r = 0; r < rows; ++r) m_fullData(r, c) = parseValue(m_rawCSVData[r][c], m_columns[c]);
        }

        calculateStatistics();

        m_sampleTypes.assign(rows, SampleType::Training);
        shuffleSampleTypes();
        m_autoNormalizeEnabled = autoNormalize;
        m_isNormalized = autoNormalize;
        if (m_autoNormalizeEnabled) normalizeData();
        return true;
    }

    /**
     * @brief Toggles whether automatic normalization is enabled.
     * @param enable True to enable, false to disable.
     */
    void setAutoNormalizeEnabled(bool enable) { m_autoNormalizeEnabled = enable; m_isNormalized = enable; }

    /**
     * @brief Checks if the dataset's internal data is currently normalized.
     * @return True if normalized, false otherwise.
     */
    bool isNormalized() const { return m_isNormalized; }

    /**
     * @brief Sets the number of columns at the end of the dataset to be treated as target variables.
     * @param num Number of target variables.
     */
    void setTargetVariableNum(int num) { m_targetVariableNum = num; }

    /**
     * @brief Gets the number of target variables currently configured.
     * @return Integer count.
     */
    int getTargetVariableNum() const { return m_targetVariableNum; }

    /** @return The original raw CSV data rows (filtered by columns kept). */
    const std::vector<std::vector<std::string>>& getOriginalCSVData() const { return m_rawCSVData; }

    /**
     * @brief Recalculates min/max statistics for all columns based on the current full data.
     */
    void calculateStatistics() {
        if (m_fullData.size() == 0) return;
        size_t rows = m_fullData.shape()[0];
        size_t cols = m_fullData.shape()[1];
        for (size_t c = 0; c < cols; ++c) {
            float minVal = 1e30f, maxVal = -1e30f;
            for (size_t r = 0; r < rows; ++r) {
                minVal = std::min(minVal, m_fullData(r, c));
                maxVal = std::max(maxVal, m_fullData(r, c));
            }
            m_columns[c].min = minVal;
            m_columns[c].max = maxVal;
        }
    }

    /**
     * @brief Resolves a float value back to its original label string if the column is categorical.
     * @param actualColIdx Index of the column.
     * @param value Float value to resolve.
     * @return Label string or string representation of the float.
     */
    std::string getLabelName(int actualColIdx, float value) const {
        if (actualColIdx < 0 || (size_t)actualColIdx >= m_columns.size()) return "";
        const auto& info = m_columns[actualColIdx];
        if (info.type != DataType::Text) return std::to_string(value);
        
        int id = (int)std::round(value);
        if (id >= 0 && (size_t)id < info.reverse_category_map.size()) {
            return info.reverse_category_map[id];
        }
        return std::to_string(value);
    }

    /**
     * @brief Maps a target variable index to its actual column index in the full data matrix.
     * @param targetVarIdx Index of the target variable (0 to targetVariableNum - 1).
     * @return Actual column index.
     */
    int getTargetColumnIndex(int targetVarIdx = 0) const {
        int inputColsBoundary = (int)m_columns.size() - m_targetVariableNum;
        return inputColsBoundary + targetVarIdx;
    }

    /**
     * @brief Configures how a specific column should be used (USED, UNUSED, or ORDERING).
     * @param name Name of the column.
     * @param usage Desired usage status.
     */
    void setColumnUsage(const std::string& name, ColumnUsage usage) {
        std::string target = trim(name);
        for (auto& col : m_columns) {
            if (trim(col.name) == target) {
                col.usage = usage;
                return;
            }
        }
    }

    /**
     * @brief Retrieves indices of columns currently marked for usage.
     * @param includeTarget If true, returns only target column indices. If false, returns only input column indices.
     * @return Vector of indices.
     */
    std::vector<int> getUsedColumnIndices(bool includeTarget = false) const {
        std::vector<int> indices;
        int inputColsBoundary = (int)m_columns.size() - m_targetVariableNum;

        if (!includeTarget) {
            int limit = (inputColsBoundary > 0 ? inputColsBoundary : 0);
            for (int i = 0; i < limit; ++i) {
                if (m_columns[i].usage == ColumnUsage::USED) indices.push_back(i);
            }
            
            // If no specific input columns are found (common in "No-Prep" forecasting),
            // we default to using target columns as inputs.
            if (indices.empty()) {
                for (int i = (inputColsBoundary > 0 ? inputColsBoundary : 0); i < (int)m_columns.size(); ++i) {
                    if (m_columns[i].usage == ColumnUsage::USED) indices.push_back(i);
                }
            }
        } else {
            for (int i = (inputColsBoundary > 0 ? inputColsBoundary : 0); i < (int)m_columns.size(); ++i) {
                indices.push_back(i);
            }
        }
        return indices;
    }

    /**
     * @brief Returns the total number of input variables (features) used for training.
     * @return Integer count.
     */
    int getInputVariableNum() const { 
        return (int)getUsedColumnIndices(false).size();
    }

    /** @return Total number of samples in the dataset. */
    size_t getSampleNum() const { return m_fullData.shape()[0]; }

    /** @return Vector containing sample category (Train/Val/Test) for each row. */
    const std::vector<SampleType>& getSampleTypes() const { return m_sampleTypes; }

    /** @return A copy of the full internal data matrix. */
    owTensor<float, 2> getData() const { return m_fullData; }

    /** @return The original name of the column at the specified index. */
    std::string getColumnName(int colIdx) const {
        if (colIdx < 0 || (size_t)colIdx >= m_columns.size()) return "Unknown";
        return m_columns[colIdx].name;
    }

    /** @return The index of the column with the given name, or -1 if not found. */
    int getColumnIndex(const std::string& name) const {
        std::string target = trim(name);
        for (size_t i = 0; i < m_columns.size(); ++i) {
            if (trim(m_columns[i].name) == target) return (int)i;
        }
        return -1;
    }

    /** @return All values in a column converted to their original string representation. */
    std::vector<std::string> getColumnAsStrings(int colIdx) const {
        std::vector<std::string> res;
        if (colIdx < 0 || (size_t)colIdx >= m_columns.size()) return res;
        size_t rows = m_fullData.shape()[0];
        res.reserve(rows);
        for (size_t r = 0; r < rows; ++r) {
            res.push_back(getLabelName(colIdx, m_fullData(r, (size_t)colIdx)));
        }
        return res;
    }

    /**
     * @brief Applies Min-Max normalization to the entire internal dataset.
     */
    void normalizeData() {
        if (m_fullData.size() == 0) return;
        calculateStatistics();
        size_t rows = m_fullData.shape()[0];
        size_t cols = m_fullData.shape()[1];
        for (size_t c = 0; c < cols; ++c) {
            if (m_columns[c].type != DataType::Numeric) continue; // Skip non-numeric columns
            
            float minVal = m_columns[c].min;
            float maxVal = m_columns[c].max;
            float range = maxVal - minVal;
            if (range == 0.0f) range = 1.0f;
            for (size_t r = 0; r < rows; ++r) {
                m_fullData(r, c) = (m_fullData(r, c) - minVal) / range;
            }
        }
        m_autoNormalizeEnabled = false; 
        m_isNormalized = true;
    }

    /**
     * @brief Normalizes an external input tensor using the dataset's statistics.
     * @param data 2D tensor to be normalized in-place.
     */
    void normalize(owTensor<float, 2>& data) const {
        std::vector<int> inputIndices = getUsedColumnIndices(false);
        if (data.shape()[1] != inputIndices.size()) return;

        for (size_t i = 0; i < data.shape()[0]; ++i) {
            for (size_t j = 0; j < inputIndices.size(); ++j) {
                int colIdx = inputIndices[j];
                float minVal = m_columns[colIdx].min;
                float maxVal = m_columns[colIdx].max;
                float range = maxVal - minVal;
                if (range == 0.0f) range = 1.0f;
                data(i, j) = (data(i, j) - minVal) / range;
            }
        }
    }

    /**
     * @brief Scales values back to their original range using target column statistics.
     * @param data 2D tensor to be de-normalized in-place.
     * @param targetVarIdx Index of the target variable to use for statistics.
     */
    void inverseNormalize(owTensor<float, 2>& data, int targetVarIdx = 0) {
        int actualColIdx = getTargetColumnIndex(targetVarIdx);
        float minV = m_columns[actualColIdx].min;
        float maxV = m_columns[actualColIdx].max;
        float range = maxV - minV;
        if (range == 0.0f) range = 1.0f;
        for (size_t i = 0; i < data.shape()[0]; ++i) {
            for (size_t j = 0; j < data.shape()[1]; ++j) {
                data(i, j) = data(i, j) * range + minV;
            }
        }
    }

    /**
     * @brief Prepares the dataset for time-series forecasting by creating lag features.
     * 
     * This method implements a sliding window at the dataset level. For each sample,
     * it prepends 'windowSize' historical values of the target column as new features.
     * 
     * @param windowSize Number of historical steps to include as features.
     * @param dilation Spacing between sampled historical steps (default 1).
     */
    /**
     * @brief Prepares the dataset for time-series forecasting by creating historical lags and future horizons.
     * 
     * This method implements a sliding window at the dataset level. For each sample,
     * it prepends 'windowSize' historical lookback values of the target column as features (lags),
     * and when 'horizon' > 1, creates multi-step future targets (ahead).
     * 
     * @param windowSize Number of historical steps to include as features (lookback).
     * @param horizon Number of future steps to predict directly (steps ahead, default 1).
     * @param dilation Spacing between sampled historical steps (default 1).
     */
    void prepareForecastData(int windowSize, int horizon = 1, int dilation = 1) {
        if (m_fullData.size() == 0 || windowSize <= 0) return;
        if (horizon < 1) horizon = 1;
        if (dilation < 1) dilation = 1;

        size_t originalRows = m_fullData.shape()[0];
        size_t originalCols = m_fullData.shape()[1];
        int referenceCol = (int)originalCols - m_targetVariableNum;
        if (referenceCol < 0 || referenceCol >= (int)originalCols) referenceCol = (int)originalCols - 1;

        size_t offset = (size_t)windowSize * (size_t)dilation;
        size_t maxLookahead = (size_t)(horizon - 1) * (size_t)dilation;
        size_t totalOffset = offset + maxLookahead;
        if (originalRows <= totalOffset) return;

        size_t newRows = originalRows - totalOffset;
        size_t newCols = (size_t)windowSize + (originalCols - (size_t)m_targetVariableNum) + (size_t)horizon;
        owTensor<float, 2> newData(newRows, newCols);
        std::vector<SampleType> newSampleTypes(newRows);

        for (size_t i = 0; i < newRows; ++i) {
            size_t actualIdx = i + offset;

            // 1. Historical Lag features (lookback: oldest to newest)
            for (int w = 0; w < windowSize; ++w) {
                size_t lookback = (size_t)(windowSize - w) * (size_t)dilation;
                newData(i, (size_t)w) = m_fullData(actualIdx - lookback, (size_t)referenceCol);
            }

            // 2. Non-target input features at current time step
            size_t colCursor = (size_t)windowSize;
            for (int j = 0; j < referenceCol; ++j) {
                newData(i, colCursor++) = m_fullData(actualIdx, (size_t)j);
            }

            // 3. Target feature(s) for horizon steps (t+1, t+2, ... t+H)
            for (int h = 0; h < horizon; ++h) {
                size_t futureIdx = actualIdx + (size_t)h * (size_t)dilation;
                newData(i, colCursor + (size_t)h) = m_fullData(futureIdx, (size_t)referenceCol);
            }

            newSampleTypes[i] = m_sampleTypes[actualIdx];
        }

        m_fullData = newData;
        m_sampleTypes = newSampleTypes;

        std::vector<ColumnInfo> newColumns;
        std::string refName = m_columns[referenceCol].name;
        float refMin = m_columns[referenceCol].min;
        float refMax = m_columns[referenceCol].max;

        // Lag columns: refName_lagW, ..., refName_lag1
        for (int w = 0; w < windowSize; ++w) {
            ColumnInfo lagCol = {refName + "_lag" + std::to_string(windowSize - w), DataType::Numeric, Ordering::Standard, ColumnUsage::USED};
            lagCol.min = refMin;
            lagCol.max = refMax;
            newColumns.push_back(lagCol);
        }

        // Non-target columns
        for (int j = 0; j < referenceCol; ++j) {
            newColumns.push_back(m_columns[j]);
        }

        // Target horizon columns
        if (horizon == 1) {
            newColumns.push_back(m_columns[referenceCol]);
        } else {
            for (int h = 0; h < horizon; ++h) {
                ColumnInfo aheadCol = {refName + "_ahead" + std::to_string(h + 1), DataType::Numeric, Ordering::Standard, ColumnUsage::USED};
                aheadCol.min = refMin;
                aheadCol.max = refMax;
                newColumns.push_back(aheadCol);
            }
        }

        m_columns = newColumns;
        m_targetVariableNum = horizon;
        m_forecastWindowSize = windowSize;
        m_forecastHorizon = horizon;
        m_forecastDilation = dilation;
    }

    /** @return Lookback window size used in prepareForecastData. */
    int getForecastWindowSize() const { return m_forecastWindowSize; }

    /** @return Forecast horizon (steps ahead) used in prepareForecastData. */
    int getForecastHorizon() const { return m_forecastHorizon; }

    /** @return Dilation factor used in prepareForecastData. */
    int getForecastDilation() const { return m_forecastDilation; }

    /**
     * @brief Retrieves the most recent (last) sample's features from the dataset.
     * @return 2D tensor [1, InputFeatures].
     */
    owTensor<float, 2> getLastSample() const {
        std::vector<int> indices = getUsedColumnIndices(false);
        if (m_fullData.shape()[0] == 0 || indices.empty()) return owTensor<float, 2>(0, 0);
        owTensor<float, 2> res(1, indices.size());
        size_t lastRow = m_fullData.shape()[0] - 1;
        for (size_t j = 0; j < indices.size(); ++j) res(0, j) = m_fullData(lastRow, (size_t)indices[j]);
        return res;
    }

    /**
     * @brief Gets min/max normalization parameters for a specific column index.
     * @param colIdx Actual index of the column in the data matrix.
     * @return Pair containing {min, max}.
     */
    std::pair<float, float> getNormalizationParamsByColumnIndex(int colIdx) const {
        if (colIdx < 0 || (size_t)colIdx >= m_columns.size()) return {0.0f, 1.0f};
        return {m_columns[colIdx].min, m_columns[colIdx].max};
    }

    /**
     * @brief Gets normalization parameters for a used column index (0 to InputVariableNum-1).
     * @param usedColIdx Relative index among columns marked as USED for input.
     * @return Pair containing {min, max}.
     */
    std::pair<float, float> getNormalizationParams(int usedColIdx) const {
        std::vector<int> indices = getUsedColumnIndices(false);
        if (usedColIdx < 0 || (size_t)usedColIdx >= indices.size()) return {0.0f, 1.0f};
        int actualIdx = indices[usedColIdx];
        return {m_columns[actualIdx].min, m_columns[actualIdx].max};
    }

    /**
     * @brief Gets normalization parameters for a column by its header name.
     * @param name Name of the column.
     * @return Pair containing {min, max}.
     */
    std::pair<float, float> getNormalizationParams(const std::string& name) const {
        for (const auto& col : m_columns) {
            if (trim(col.name) == trim(name)) return {col.min, col.max};
        }
        return {0.0f, 1.0f};
    }

    /**
     * @brief Low-level filtering method to extract specific sample types and roles.
     * @param targetType Desired SampleType (Training, Validation, Test).
     * @param isInput If true, returns features. If false, returns targets.
     * @return Filtered data matrix.
     */
    owTensor<float, 2> getRowsAndColsFiltered(SampleType targetType, bool isInput) const {
        std::vector<int> colIndices = getUsedColumnIndices(!isInput);
        size_t rows = 0;
        for (auto t : m_sampleTypes) if (t == targetType) rows++;
        if (rows == 0) return owTensor<float, 2>(0, colIndices.size());
        owTensor<float, 2> res(rows, colIndices.size());
        size_t curr = 0;
        for (size_t i = 0; i < m_sampleTypes.size(); ++i) {
            if (m_sampleTypes[i] == targetType) {
                for (size_t j = 0; j < colIndices.size(); ++j) {
                    int colIdx = colIndices[j];
                    float val = m_fullData(i, (size_t)colIdx);
                    if (m_autoNormalizeEnabled && m_columns[colIdx].usage == ColumnUsage::USED) {
                        float minV = m_columns[colIdx].min;
                        float maxV = m_columns[colIdx].max;
                        float range = maxV - minV;
                        if (range == 0) range = 1.0f;
                        val = (val - minV) / range;
                    }
                    res(curr, j) = val;
                }
                curr++;
            }
        }
        return res;
    }

    /** @return Features for samples in the Training set. */
    owTensor<float, 2> getTrainInput() const { return getRowsAndColsFiltered(SampleType::Training, true); }
    /** @return Targets for samples in the Training set. */
    owTensor<float, 2> getTrainTarget() const { return getRowsAndColsFiltered(SampleType::Training, false); }
    /** @return Features for samples in the Validation set. */
    owTensor<float, 2> getValInput() const { return getRowsAndColsFiltered(SampleType::Validation, true); }
    /** @return Targets for samples in the Validation set. */
    owTensor<float, 2> getValTarget() const { return getRowsAndColsFiltered(SampleType::Validation, false); }
    /** @return Features for samples in the Test set. */
    owTensor<float, 2> getTestInput() const { return getRowsAndColsFiltered(SampleType::Test, true); }
    /** @return Targets for samples in the Test set. */
    owTensor<float, 2> getTestTarget() const { return getRowsAndColsFiltered(SampleType::Test, false); }

    /**
     * @brief Retrieves the input feature vector for a specific sample index.
     * @param sampleNo Index of the sample.
     * @return 1D tensor containing only columns marked as USED for input.
     */
    owTensor<float, 1> getInputValues(size_t sampleNo) const {
        std::vector<int> inputIndices = getUsedColumnIndices(false);
        if (sampleNo >= m_fullData.shape()[0]) return owTensor<float, 1>(0);
        owTensor<float, 1> res(inputIndices.size());
        for (size_t j = 0; j < inputIndices.size(); ++j) {
            res(j) = m_fullData(sampleNo, (size_t)inputIndices[j]);
        }
        return res;
    }

    /**
     * @brief Retrieves the target vector for a specific sample index.
     * @param sampleNo Index of the sample.
     * @return 1D tensor containing the target column(s).
     */
    owTensor<float, 1> getTargetValues(size_t sampleNo) const {
        std::vector<int> targetIndices = getUsedColumnIndices(true);
        if (sampleNo >= m_fullData.shape()[0]) return owTensor<float, 1>(0);
        owTensor<float, 1> res(targetIndices.size());
        for (size_t j = 0; j < targetIndices.size(); ++j) {
            res(j) = m_fullData(sampleNo, (size_t)targetIndices[j]);
        }
        return res;
    }

    /** @return All input features for the entire dataset. */
    owTensor<float, 2> getAllInput() const { return getFullDataFiltered(true); }
    /** @return All targets for the entire dataset. */
    owTensor<float, 2> getAllTarget() const { return getFullDataFiltered(false); }

    /**
     * @brief Retrieves all data for either input or target roles, applying auto-normalization if enabled.
     * @param isInput True for features, false for targets.
     * @return Matrix containing requested data.
     */
    owTensor<float, 2> getFullDataFiltered(bool isInput) const {
        std::vector<int> colIndices = getUsedColumnIndices(!isInput);
        size_t rows = m_fullData.shape()[0];
        owTensor<float, 2> res(rows, colIndices.size());
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < colIndices.size(); ++j) {
                int colIdx = colIndices[j];
                float val = m_fullData(i, (size_t)colIdx);
                if (m_autoNormalizeEnabled && m_columns[colIdx].usage == ColumnUsage::USED) {
                    float minV = m_columns[colIdx].min;
                    float maxV = m_columns[colIdx].max;
                    float range = maxV - minV;
                    if (range == 0) range = 1.0f;
                    val = (val - minV) / range;
                }
                res(i, j) = val;
            }
        }
        return res;
    }

    /**
     * @brief Configures dataset splitting ratios and optionally shuffles sample assignments.
     * @param train Ratio for training data (0.0 - 1.0).
     * @param val Ratio for validation data.
     * @param test Ratio for test data.
     * @param shuffle If true, shuffles sample type assignments.
     */
    void setRatios(float train, float val, float test, bool shuffle = true) {
        m_trainRatio = train; m_valRatio = val; m_testRatio = test;
        shuffleSampleTypes(shuffle);
    }

    /** @param d Character to use as CSV delimiter (e.g., ',', ';'). */
    void setDelimiter(char d) { m_delimiter = d; }
    /** @return Currently configured delimiter. */
    char getDelimiter() const { return m_delimiter; }

    /**
     * @brief Randomly assigns samples to Training, Validation, and Test sets based on configured ratios.
     * @param shuffle If true, uses the internal RNG to randomize assignments.
     */
    void shuffleSampleTypes(bool shuffle = true) {
        if (m_sampleTypes.empty()) return;
        size_t rows = m_sampleTypes.size();
        size_t trainCount = (size_t)(rows * m_trainRatio);
        size_t valCount = (size_t)(rows * m_valRatio);
        std::vector<SampleType> newTypes(rows);
        for (size_t i = 0; i < rows; ++i) {
            if (i < trainCount) newTypes[i] = SampleType::Training;
            else if (i < trainCount + valCount) newTypes[i] = SampleType::Validation;
            else newTypes[i] = SampleType::Test;
        }
        if (shuffle) {
            std::shuffle(newTypes.begin(), newTypes.end(), m_rng);
        }
        m_sampleTypes = newTypes;
    }

    /**
     * @brief Returns a string representation of a sample's set assignment.
     * @param index Row index.
     * @return "Training", "Validation", "Testing", or "Unknown".
     */
    std::string getSampleTypeString(size_t index) const {
        if (index >= m_sampleTypes.size()) return "Unknown";
        if (m_sampleTypes[index] == SampleType::Training) return "Training";
        if (m_sampleTypes[index] == SampleType::Validation) return "Validation";
        return "Testing";
    }

    /**
     * @brief Exports the current state of the dataset to a CSV file.
     * @param filepath Path to the output CSV file.
     * @param dateColumnName Optional: Name of the column to use as the first Date column.
     */
    bool saveToCSV(const std::string& filepath, const std::string& dateColumnName = "") const {
        if (m_fullData.size() == 0) return false;
        std::ofstream file(filepath);
        if (!file.is_open()) return false;

        std::vector<int> usedIndices;
        for (int i = 0; i < (int)m_columns.size(); ++i) {
            // If it's the date column and we're prefixing with it, don't include it in the data section
            if (!dateColumnName.empty() && trim(m_columns[i].name) == trim(dateColumnName)) continue;
            if (m_columns[i].usage == ColumnUsage::USED) {
                usedIndices.push_back(i);
            }
        }

        // 1. Write Header
        if (!dateColumnName.empty()) {
            file << dateColumnName << m_delimiter;
        }
        for (size_t i = 0; i < usedIndices.size(); ++i) {
            file << m_columns[usedIndices[i]].name << (i == usedIndices.size() - 1 ? "" : std::string(1, m_delimiter));
        }
        file << "\n";

        // 2. Write Data
        size_t rows = m_fullData.shape()[0];
        std::vector<std::string> dates;
        if (!dateColumnName.empty()) {
            int dateIdx = getColumnIndex(dateColumnName);
            if (dateIdx != -1) dates = getColumnAsStrings(dateIdx);
        }

        for (size_t r = 0; r < rows; ++r) {
            if (!dates.empty() && r < dates.size()) {
                file << dates[r] << m_delimiter;
            }
            for (size_t i = 0; i < usedIndices.size(); ++i) {
                int colIdx = usedIndices[i];
                file << std::fixed << std::setprecision(6) << m_fullData(r, (size_t)colIdx) << (i == usedIndices.size() - 1 ? "" : std::string(1, m_delimiter));
            }
            file << "\n";
        }
        file.close();
        return true;
    }

private:
    owTensor<float, 2> m_fullData;
    std::vector<std::vector<std::string>> m_rawCSVData;
    std::vector<ColumnInfo> m_columns;
    std::vector<SampleType> m_sampleTypes;
    int m_targetVariableNum = 1;
    int m_forecastWindowSize = 0;
    int m_forecastHorizon = 1;
    int m_forecastDilation = 1;
    float m_trainRatio = 0.6f, m_valRatio = 0.2f, m_testRatio = 0.2f;
    char m_delimiter = ';'; 
    bool m_autoNormalizeEnabled = false;
    bool m_isNormalized = false;
    std::mt19937 m_rng;

    float parseValue(const std::string& val, ColumnInfo& info) {
        if (info.type == DataType::Text) {
            auto it = info.category_map.find(val);
            if (it != info.category_map.end()) return it->second;
            return 0.0f;
        }
        if (val.empty()) return 0.0f;
        std::string s = val;
        std::replace(s.begin(), s.end(), ',', '.');
        try {
            return std::stof(s);
        } catch (...) {
            return (float)std::atof(s.c_str());
        }
    }

    std::string trim(const std::string& s) const {
        size_t first = s.find_first_not_of(" \t\r\n\xEF\xBB\xBF");
        if (first == std::string::npos) return "";
        size_t last = s.find_last_not_of(" \t\r\n");
        return s.substr(first, (last - first + 1));
    }

    std::string cleanColumnName(const std::string& name) const {
        return trim(name);
    }
};

} // namespace ow
