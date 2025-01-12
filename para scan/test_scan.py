import torch
from torch.utils.cpp_extension import load
import os

module_path = os.path.dirname(__file__)

# Load the extension
parallel_scan_module = load(
    name="parallel_scan2",
    sources=["scan_wrapper.cpp", "scan_kernel.cu"],
    verbose=True  # Set this to True for helpful compilation output
)

B, D, L = 16, 64, 1024
input_tensor = torch.rand((B, D, L), device='cuda:0')
output_tensor = torch.zeros((B, D, L), device='cuda:0')

# Run the parallel scan
parallel_scan_module.parallel_scan(output_tensor, input_tensor, B, D, L)

# Compare results with cumsum
output_tensor2 = input_tensor.cumsum(-1)

# Check if the outputs match
if not torch.allclose(output_tensor, output_tensor2, atol=1e-5):
    print("Mismatch between parallel scan and cumsum results!")
    # Check the first 2 elements
    print("Parallel Scan Output:", output_tensor[:2])
    print("Cumsum Output:", output_tensor2[:2])
else:
    print("Results match!")

breakpoint()
