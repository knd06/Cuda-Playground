#include <torch/types.h>


__global__ void parallel_scan_kernel(float* d_out, const float* d_in, int B, int D, int L) {
    __shared__ float warp_sums[32];

    int tid = threadIdx.x;
    int laneId = tid & 31;
    int warpId = tid >> 5;
    int b = blockIdx.x;
    int d = blockIdx.y;
    int global_idx = ((b * D + d) * L) + tid;

    // Bounds check
    if (tid >= L) return;

    float value = (tid < L) ? d_in[global_idx] : 0.0f;

    // Warp-level scan
    for (int offset = 1; offset < 32; offset *= 2) {
        float up_value = __shfl_up_sync(0xffffffff, value, offset);
        if (laneId >= offset) {
            value += up_value;
        }
    }

    // Store last thread's result into shared memory
    if (laneId == 31) {
        warp_sums[warpId] = value;
    }
    __syncthreads();

    // Scan warp sums with the first warp
    if (warpId == 0) {
        float acc_value = warp_sums[laneId];
        for (int offset = 1; offset < 32; offset *= 2) {
            float up_value = __shfl_up_sync(0xffffffff, acc_value, offset);
            if (laneId >= offset) {
                acc_value += up_value;
            }
        }
        warp_sums[laneId] = acc_value;
    }
    __syncthreads();

    // Add warp offset to values
    if (warpId != 0) {
        value += warp_sums[warpId - 1];
    }

    // Write results back
    if (tid < L) {
        d_out[global_idx] = value;
    }
}


void parallel_scan(torch::Tensor d_out, torch::Tensor d_in, int B, int D, int L) {
    const dim3 blockSize(1024, 1, 1);
    const dim3 gridSize(B, D, 1);
    parallel_scan_kernel<<<gridSize, blockSize>>>(d_out.data_ptr<float>(), d_in.data_ptr<float>(), B, D, L);
}
