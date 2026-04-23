#include "owCuda.hpp"
#include <device_launch_parameters.h>

#ifdef OW_USE_GPU

namespace ow {
namespace cuda {

/**
 * @brief Optimized Tiled Matrix Multiplication Kernel
 */
template <int TILE_SIZE>
__global__ void linearForwardTiledKernel(const float* A, const float* B, const float* bias, float* C, 
                                        int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < K && t * TILE_SIZE + threadIdx.y < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * K + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = sum + bias[col];
    }
}

__global__ void matMulKernel(const float* A, const float* B, float* C, int M, int N, int K, bool transA, bool transB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            float valA = transA ? A[i * M + row] : A[row * N + i];
            float valB = transB ? B[col * N + i] : B[i * K + col];
            sum += valA * valB;
        }
        C[row * K + col] = sum;
    }
}

// --- Activation Kernels ---
__global__ void reluForwardKernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = (data[i] > 0.0f) ? data[i] : 0.0f;
}
__global__ void reluBackwardKernel(float* grad, const float* input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] *= (input[i] > 0.0f) ? 1.0f : 0.0f;
}
__global__ void sigmoidForwardKernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = 1.0f / (1.0f + expf(-data[i]));
}
__global__ void sigmoidBackwardKernel(float* grad, const float* input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = 1.0f / (1.0f + expf(-input[i]));
        grad[i] *= s * (1.0f - s);
    }
}
__global__ void tanhForwardKernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = tanhf(data[i]);
}
__global__ void tanhBackwardKernel(float* grad, const float* input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float t = tanhf(input[i]);
        grad[i] *= (1.0f - t * t);
    }
}

// --- Loss & Utility Kernels ---
__global__ void mseLossKernel(const float* pred, const float* target, float* loss, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float diff = pred[i] - target[i];
        atomicAdd(loss, (diff * diff) / (float)n);
    }
}
__global__ void mseGradientKernel(const float* pred, const float* target, float* grad, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] = 2.0f * (pred[i] - target[i]) / (float)n;
}
__global__ void biasGradientKernel(const float* dz, float* db, int batchSize, int outputSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < outputSize) {
        float sum = 0.0f;
        for (int b = 0; b < batchSize; ++b) sum += dz[b * outputSize + col];
        db[col] = sum;
    }
}
__global__ void adamUpdateKernel(float* params, float* grads, float* m, float* v, int n, float lr, float b1, float b2, float eps, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        m[i] = b1 * m[i] + (1.0f - b1) * grads[i];
        v[i] = b2 * v[i] + (1.0f - b2) * grads[i] * grads[i];
        float m_hat = m[i] / (1.0f - powf(b1, (float)t));
        float v_hat = v[i] / (1.0f - powf(b2, (float)t));
        params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}
__global__ void regularizationKernel(float* weights, float* grads, int n, int type, float lambda) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (type == 1) grads[i] += lambda * weights[i];
        else if (type == 2) grads[i] += lambda * (weights[i] > 0.0f ? 1.0f : -1.0f);
    }
}

// --- L-BFGS Double Precision Kernels ---
__global__ void vecAddScaledKernel(double* res, const double* a, const double* b, double scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) res[i] = a[i] + scale * b[i];
}
__global__ void vecDotKernel(const double* a, const double* b, double* res, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(res, a[i] * b[i]);
}
__global__ void vecScaleKernel(double* data, double scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= scale;
}
__global__ void castFloatToDoubleKernel(double* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (double)src[i];
}
__global__ void castDoubleToFloatKernel(float* dst, const double* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (float)src[i];
}

// --- Wrappers (EVERYTHING MUST SYNC WHEN CPU NEEDS IT) ---
void linearForward(const float* i, const float* w, const float* b, float* o, int bs, int is, int os) {
    linearForwardTiledKernel<16><<<dim3((os+15)/16,(bs+15)/16),dim3(16,16)>>>(i,w,b,o,bs,is,os);
    cudaDeviceSynchronize();
}
void matMul(const float* A, const float* B, float* C, int M, int N, int K, bool tA, bool tB) {
    matMulKernel<<<dim3((K+15)/16, (M+15)/16), dim3(16,16)>>>(A, B, C, M, N, K, tA, tB);
    cudaDeviceSynchronize();
}
void reluForward(float* d, int s) { reluForwardKernel<<<(s+255)/256,256>>>(d,s); cudaDeviceSynchronize(); }
void reluBackward(float* g, const float* i, int s) { reluBackwardKernel<<<(s+255)/256,256>>>(g,i,s); cudaDeviceSynchronize(); }
void sigmoidForward(float* d, int s) { sigmoidForwardKernel<<<(s+255)/256,256>>>(d,s); cudaDeviceSynchronize(); }
void sigmoidBackward(float* g, const float* i, int s) { sigmoidBackwardKernel<<<(s+255)/256,256>>>(g,i,s); cudaDeviceSynchronize(); }
void tanhForward(float* d, int s) { tanhForwardKernel<<<(s+255)/256,256>>>(d,s); cudaDeviceSynchronize(); }
void tanhBackward(float* g, const float* i, int s) { tanhBackwardKernel<<<(s+255)/256,256>>>(g,i,s); cudaDeviceSynchronize(); }

void mseLoss(const float* p, const float* t, float* l, int s) {
    cudaMemset(l, 0, sizeof(float));
    mseLossKernel<<<(s+255)/256, 256>>>(p, t, l, s);
    cudaDeviceSynchronize();
}
void mseGradient(const float* p, const float* t, float* g, int s) { mseGradientKernel<<<(s+255)/256, 256>>>(p, t, g, s); cudaDeviceSynchronize(); }
void computeBiasGradient(const float* dz, float* db, int bs, int os) { biasGradientKernel<<<(os+255)/256, 256>>>(dz, db, bs, os); cudaDeviceSynchronize(); }
void adamUpdate(float* p, float* g, float* m, float* v, int s, float lr, float b1, float b2, float e, int t) { adamUpdateKernel<<<(s+255)/256, 256>>>(p, g, m, v, s, lr, b1, b2, e, t); cudaDeviceSynchronize(); }
void applyRegularization(float* w, float* g, int s, int rt, float l) { regularizationKernel<<<(s+255)/256, 256>>>(w, g, s, rt, l); cudaDeviceSynchronize(); }

void vecAddScaled(double* res, const double* a, const double* b, double scale, int n) { vecAddScaledKernel<<<(n+255)/256, 256>>>(res, a, b, scale, n); cudaDeviceSynchronize(); }
void vecDot(const double* a, const double* b, double* res, int n) { cudaMemset(res, 0, sizeof(double)); vecDotKernel<<<(n+255)/256, 256>>>(a, b, res, n); cudaDeviceSynchronize(); }
void vecCopy(double* dst, const double* src, int n) { cudaMemcpy(dst, src, n * sizeof(double), cudaMemcpyDeviceToDevice); cudaDeviceSynchronize(); }
void vecScale(double* data, double scale, int n) { vecScaleKernel<<<(n+255)/256, 256>>>(data, scale, n); cudaDeviceSynchronize(); }
void castFloatToDouble(double* dst, const float* src, int n) { castFloatToDoubleKernel<<<(n+255)/256, 256>>>(dst, src, n); cudaDeviceSynchronize(); }
void castDoubleToFloat(float* dst, const double* src, int n) { castDoubleToFloatKernel<<<(n+255)/256, 256>>>(dst, src, n); cudaDeviceSynchronize(); }

} // namespace cuda
} // namespace ow

#endif
