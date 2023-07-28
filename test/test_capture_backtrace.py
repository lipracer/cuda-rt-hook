import torch
import cuda_mock

cuda_mock.internal_install_hook("libc.so", "libtorch_python.so", "strlen")

a = torch.rand(2, 3)
b = torch.rand(2, 3)
c = torch.add(a, 1)
print(c)