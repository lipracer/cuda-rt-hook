import torch
import cuda_mock

cuda_mock.internal_install_hook("libtorch_cpu.so", "libtorch_python.so", "_ZN2at6native23NestedTensor_add_TensorERKNS_6TensorES3_RKN3c106ScalarE")
cuda_mock.internal_install_hook("libtorch_cpu.so", "libtorch_python.so", "_ZN2at6native14mkldnn_add_outERKNS_6TensorES3_RKN3c106ScalarERS1_")
cuda_mock.internal_install_hook("libtorch_cpu.so", "libtorch_python.so", "_ZN2at6native3dotERKNS_6TensorES3_")

a = torch.rand(2, 3)
b = torch.rand(2, 3)
c = torch.add(a, b)
print(c)

lhs = torch.rand(2)
rhs = torch.rand(2,)
c = torch.dot(lhs, rhs)
print(c)