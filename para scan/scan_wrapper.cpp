#include <torch/extension.h>


void parallel_scan(torch::Tensor d_out, torch::Tensor d_in, int B, int D, int L);


void validate_tensors(const torch::Tensor& d_out, const torch::Tensor& d_in, int B, int D, int L) {
    TORCH_CHECK(d_out.is_cuda(), "Output tensor must be a CUDA tensor");
    TORCH_CHECK(d_in.is_cuda(), "Input tensor must be a CUDA tensor");

    TORCH_CHECK(d_out.scalar_type() == torch::kFloat, "Output tensor must be float");
    TORCH_CHECK(d_in.scalar_type() == torch::kFloat, "Input tensor must be float");

    TORCH_CHECK(d_out.is_contiguous(), "Output tensor must be contiguous");
    TORCH_CHECK(d_in.is_contiguous(), "Input tensor must be contiguous");

    TORCH_CHECK(d_in.size(0) == B, "Input tensor batch size mismatch");
    TORCH_CHECK(d_in.size(1) == D, "Input tensor depth mismatch");
    TORCH_CHECK(d_in.size(2) == L, "Input tensor length mismatch");
    
    TORCH_CHECK(d_out.sizes() == d_in.sizes(), "Output and input tensor sizes must match");
}

// Wrapper Function
void parallel_scan_wrapper(torch::Tensor d_out, torch::Tensor d_in, int B, int D, int L) {
    validate_tensors(d_out, d_in, B, D, L);
    parallel_scan(d_out, d_in, B, D, L);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("parallel_prefix_scan", &parallel_scan_wrapper, "Parallel Prefix Scan (CUDA)");
}
