import os
import ctypes

script_dir = os.path.dirname(os.path.abspath(__file__))
cuda_mock_impl = ctypes.CDLL(f'{script_dir}/libcuda_mock_impl.so')

def add(lhs, rhs):
    return lhs + rhs

def initialize():
    return cuda_mock_impl.initialize()

def internal_install_hook(*args):
    new_args = [ctypes.c_char_p(arg.encode('utf-8')) for arg in args]
    return cuda_mock_impl.internal_install_hook(*new_args)

def xpu_initialize():
    return cuda_mock_impl.xpu_initialize()

