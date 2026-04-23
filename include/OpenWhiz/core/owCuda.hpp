/*
 * owCuda.hpp
 *
 *  Created on: Apr 21, 2026
 *      Author: Noyan Culum, AITIAL
 */

#pragma once

#ifdef OW_USE_GPU
#include <cuda_runtime.h>
#include <iostream>

namespace ow {
namespace cuda {

/**
 * @brief Wrapper for Linear Layer Forward Pass on GPU via CUDA
 * Implementation in owCuda.cu
 */
void linearForward(const float* input, const float* weights, const float* bias, float* output,
                  int batchSize, int inputSize, int outputSize);

void matMul(const float* A, const float* B, float* C, int M, int N, int K, 
            bool transA = false, bool transB = false);

// Activations
void reluForward(float* data, int size);
void reluBackward(float* grad, const float* input, int size);
void sigmoidForward(float* data, int size);
void sigmoidBackward(float* grad, const float* output, int size); // Note: using output for sigmoid grad optimization
void tanhForward(float* data, int size);
void tanhBackward(float* grad, const float* output, int size);

// Losses (MSE example)
void mseLoss(const float* pred, const float* target, float* loss, int size);
void mseGradient(const float* pred, const float* target, float* grad, int size);
void computeBiasGradient(const float* dz, float* db, int batchSize, int outputSize);

// Optimizers
void adamUpdate(float* params, float* grads, float* m, float* v, 
                int size, float lr, float beta1, float beta2, float epsilon, int t);

void applyRegularization(float* weights, float* grads, int size, int regType, float lambda);

// Double Precision Ops for L-BFGS
void vecAddScaled(double* res, const double* a, const double* b, double scale, int n);
void vecDot(const double* a, const double* b, double* res, int n);
void vecCopy(double* dst, const double* src, int n);
void vecScale(double* data, double scale, int n);
void castFloatToDouble(double* dst, const float* src, int n);
void castDoubleToFloat(float* dst, const double* src, int n);

} // namespace cuda
} // namespace ow

#endif
