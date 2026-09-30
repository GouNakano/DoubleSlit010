#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <cmath>
#include "TDoubleSlit.h"


// 乱数初期化カーネル
__global__ void init_rand_kernel(curandState* state, unsigned long seed, int num_electrons) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_electrons) 
    {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

// 電子の衝突位置を計算するカーネル
__global__ void sample_electrons_kernel(int* d_counts, curandState* state, int num_electrons, int width, float d, float w) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_electrons)
    {
        return;
    }
    curandState local_state = state[idx];
    float pi = 3.1415926535f;

    while (true) 
    {
        float x = (curand_uniform(&local_state) - 0.5f) * width;

        float interference = cosf(pi * d * x / 800.0f);
        float diffraction = (x == 0.0f) ? 1.0f : sinf(pi * w * x / 800.0f) / (pi * w * x / 800.0f);
        float intensity = (interference * interference) * (diffraction * diffraction);

        if (curand_uniform(&local_state) < intensity) 
        { 
            int screen_x = static_cast<int>(x + width / 2.0f);
            if (screen_x >= 0 && screen_x < width) 
            {
                atomicAdd(&d_counts[screen_x], 1);
            }
            break;
        }
    }
    state[idx] = local_state;
}

//---------------------------------------------
// C++（main.cpp）から呼ぶためのラッパー関数
//---------------------------------------------

void TDoubleSlit::run_cuda_simulation(int* h_counts, int width, int num_electrons, float d, float w, unsigned long seed)
{

    // 初回実行時のみGPUメモリ確保
    if (d_rand_states == nullptr) 
    {
        cudaMalloc((void**)&d_rand_states, num_electrons * sizeof(curandState));

        init_rand_kernel<<<(num_electrons + 255) / 256, 256>>>((curandState *)d_rand_states, seed, num_electrons);
    }
    if (d_counts == nullptr || last_width != width) 
    {
        if (d_counts != nullptr)
        {
            cudaFree(d_counts);
        }
        cudaMalloc((void**)&d_counts, width * sizeof(int));
        cudaMemset(d_counts, 0, width * sizeof(int));
        last_width = width;
    }

    // カウント配列を毎フレームのインクリメント用にホストから同期（リセット対応のため）
    cudaMemcpy(d_counts, h_counts, width * sizeof(int), cudaMemcpyHostToDevice);

    // カーネル実行(※<<< >>> に近い所では、コメントの後に１行開けること！ nvcc構文解析のバグ回避)
    // <<< >>> は勝手に < << > >> に変わることがあるので、ソース修正時には常に監視すること。 (Visual Studio のソースコードの構文自動修正の余計な処理対策)

    int threads = 256;
    int blocks = (num_electrons + threads - 1) / threads;
    sample_electrons_kernel<<<blocks, threads>>>(d_counts, (curandState*)d_rand_states, num_electrons, width, d, w);

    // 計算結果をCPU（OpenCV側）に回収
    cudaMemcpy(h_counts, d_counts, width * sizeof(int), cudaMemcpyDeviceToHost);
}

//---------------------------------------------
//CUDA終了時の後処理（GPUメモリ解放）
//---------------------------------------------

void TDoubleSlit::end_cuda_simulation()
{
    cudaFree(d_rand_states);
    d_rand_states = nullptr;
    cudaFree(d_counts);
    d_counts = nullptr;

}
