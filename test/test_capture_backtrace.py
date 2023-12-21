import torch
import cuda_mock
from cuda_mock import *

lib = DynamicObj('/home/workspace/cuda-rt-hook/test/test_hook_strlen.cnn').compile().appen_compile_opts('-lpthread').get_lib()

cuda_mock.internal_install_hook("libc.so", "cuda_mock_impl.cpython-38-x86_64-linux-gnu.so", "strlen", str(lib), "strlen")
