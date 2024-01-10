import os
import ctypes
from .dynamic_obj import *

script_dir = os.path.dirname(os.path.abspath(__file__))
cuda_mock_impl = ctypes.CDLL(f'{script_dir}/libcuda_mock_impl.so')

def add(lhs, rhs):
    return lhs + rhs

def initialize():
    return cuda_mock_impl.initialize()

def uninitialize():
    return cuda_mock_impl.uninitialize()

def internal_install_hook(*args):
    new_args = [ctypes.c_char_p(arg.encode('utf-8')) for arg in args]
    return cuda_mock_impl.internal_install_hook(*new_args)

def internal_install_hook_regex(*args):
    new_args = [ctypes.c_char_p(arg.encode('utf-8')) for arg in args]
    return cuda_mock_impl.internal_install_hook_regex(*new_args)

def xpu_initialize():
    return cuda_mock_impl.xpu_initialize()

str_code = '''
#include <stddef.h>
#include <stdio.h>
#include <unistd.h>

#include <fstream>
#include <iostream>

#include "logger/logger.h"
#include "profile/Timer.h"

#define EXPORT __attribute__((__visibility__("default")))

extern "C" {{

void* __origin_{0} = nullptr;
void* __origin_{1} = nullptr;
void* __origin_{2} = nullptr;


EXPORT int {0}(const void* arg, uint64_t size, uint64_t offset) {{
    hook::Timer timer(__func__);
    int ret = reinterpret_cast<decltype(&{0})>(__origin_{0})(arg, size, offset);
    return ret;
}}

EXPORT int {1}(void* func) {{
    hook::Timer timer(__func__);
    int ret = reinterpret_cast<decltype(&{1})>(__origin_{1})(func);
    return ret;
}}

EXPORT int {2}(int nclusters, int ncores, void* stream) {{
    hook::Timer timer(__func__);
    int ret = reinterpret_cast<decltype(&{2})>(__origin_{2})(nclusters, ncores, stream);
    return ret;
}}

}}
'''

def xpu_runtime_profile():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    funcs = ["xpu_launch_argument_set", "xpu_launch_async", "xpu_launch_config"]
    lib = dynamic_obj(str_code.format(*funcs), True).appen_compile_opts('-lpthread').compile().get_lib()
    internal_install_hook_regex(r"libxpurt\.so\..*", r"[^(libxpurt\.so\..*)]", "xpu_launch_argument_set", str(lib), "xpu_launch_argument_set")
    internal_install_hook_regex(r"libxpurt\.so\..*", r"[^(libxpurt\.so\..*)]", "xpu_launch_async", str(lib), "xpu_launch_async")
    internal_install_hook_regex(r"libxpurt\.so\..*", r"[^(libxpurt\.so\..*)]", "xpu_launch_config", str(lib), "xpu_launch_config")


