#include <torch/extension.h>


void parallel_scan(torch::Tensor d_out, torch::Tensor d_in, torch::Tensor d_gate, int B, int D, int L);


void validate_tensors(const torch::Tensor& d_out, const torch::Tensor& d_in, const torch::Tensor& d_gate, int B, int D, int L) {
    TORCH_CHECK(d_out.is_cuda(), "Output tensor must be a CUDA tensor");
    TORCH_CHECK(d_in.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(d_gate.is_cuda(), "Gate tensor must be a CUDA tensor");

    TORCH_CHECK(d_out.scalar_type() == torch::kFloat, "Output tensor must be float");
    TORCH_CHECK(d_in.scalar_type() == torch::kFloat, "Input tensor must be float");
    TORCH_CHECK(d_gate.scalar_type() == torch::kFloat, "Gate tensor must be float");

    TORCH_CHECK(d_out.is_contiguous(), "Output tensor must be contiguous");
    TORCH_CHECK(d_in.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(d_gate.is_contiguous(), "Gate tensor must be contiguous");

    TORCH_CHECK(d_in.size(0) == B, "Input tensor batch size mismatch");
    TORCH_CHECK(d_in.size(1) == D, "Input tensor depth mismatch");
    TORCH_CHECK(d_in.size(2) == L, "Input tensor length mismatch");

    TORCH_CHECK(d_out.sizes() == d_in.sizes(), "Output and input tensor sizes must match");
    TORCH_CHECK(d_gate.sizes() == d_in.sizes(), "Gate tensor size must match input tensor size");
}

// Wrapper function
void parallel_scan_wrapper(torch::Tensor d_out, torch::Tensor d_in, torch::Tensor d_gate, int B, int D, int L) {
    validate_tensors(d_out, d_in, d_gate, B, D, L);
    parallel_scan(d_out, d_in, d_gate, B, D, L);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("parallel_scan", &parallel_scan_wrapper, "Parallel Prefix Scan with Gate (CUDA)");
}
