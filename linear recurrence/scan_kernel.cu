#include <torch/types.h>
#include <cuda_runtime.h>
#include <iostream>



__global__ void parallel_scan_kernel(float* d_out, const float* d_in, const float* d_gate, int B, int D, int L) {
    __shared__ float warp_sums[32];         // Shared memory for warp sums
    __shared__ float warp_prods_gate[32];  // Shared memory for warp gate products

    int tid = threadIdx.x;           // Thread index within the block
    int laneId = tid & 31;           // Lane ID within the warp
    int warpId = tid >> 5;           // Warp ID within the block
    int b = blockIdx.x;              // Batch index
    int d = blockIdx.y;              // Feature dimension index

    int global_idx = ((b * D + d) * L) + tid;

    float value = (tid < L) ? d_in[global_idx] : 0.0f;
    float gate = (tid < L) ? d_gate[global_idx] : 0.0f;

    // Compute inclusive scan within the warp
    for (int offset = 1; offset < 32; offset *= 2) {
        float up_value = __shfl_up_sync(0xffffffff, value, offset);
        float up_gate = __shfl_up_sync(0xffffffff, gate, offset);

        if (laneId >= offset) {
            value = gate * up_value + value;
            gate = gate * up_gate;
        }
    }

    // Store warp results into shared memory
    if (laneId == 31) {
        warp_sums[warpId] = value;
        warp_prods_gate[warpId] = gate;
    }

    __syncthreads();

    // Perform scan of warp-level sums with the first warp
    if (warpId == 0) {
        float acc_value = warp_sums[laneId];
        float acc_gate = warp_prods_gate[laneId];

        for (int offset = 1; offset < 32; offset *= 2) {
            float up_value = __shfl_up_sync(0xffffffff, acc_value, offset);
            float up_gate = __shfl_up_sync(0xffffffff, acc_gate, offset);

            if (laneId >= offset) {
                acc_value = acc_gate * up_value + acc_value;
                acc_gate = acc_gate * up_gate;
            }
        }

        warp_sums[laneId] = acc_value;
        warp_prods_gate[laneId] = acc_gate;
    }

    __syncthreads();

    // Accumulate warp results for final output
    if (warpId != 0) {
        value += warp_sums[warpId - 1] * gate;
    }

    // Write the result back to global memory
    if (tid < L) {
        d_out[global_idx] = value;
    }
}


void parallel_scan(torch::Tensor d_out, torch::Tensor d_in, torch::Tensor d_gate, int B, int D, int L) {
    const int maxThreadsPerBlock = 1024;
    int threads = std::min(L, maxThreadsPerBlock);
    const dim3 blockSize(threads, 1, 1);
    const dim3 gridSize(B, D, 1);

    parallel_scan_kernel<<<gridSize, blockSize>>>(
        d_out.data_ptr<float>(),
        d_in.data_ptr<float>(),
        d_gate.data_ptr<float>(),
        B, D, L
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA kernel failed.");
    }
}
