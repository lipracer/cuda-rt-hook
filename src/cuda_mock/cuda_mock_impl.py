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
    lib = ctypes.CDLL(args[-2])
    new_args = [ctypes.c_char_p(arg.encode('utf-8')) for arg in args]
    cuda_mock_impl.internal_install_hook_regex(*new_args)
    return lib

def xpu_initialize(use_improve=False):
    return cuda_mock_impl.xpu_initialize(ctypes.c_bool(use_improve))

def patch_runtime():
    return cuda_mock_impl.patch_runtime()

def log(*args):
    new_args = [ctypes.c_char_p(arg.encode('utf-8')) for arg in args]
    return cuda_mock_impl.py_log(*new_args)

class __XpuRuntimeProfiler:
    def __init__(self):
        xpu_initialize(True)
    def start_capture(self):
        cuda_mock_impl.start_capture()

    def end_capture(self):
        c_python_object = ctypes.py_object(self)
        cuda_mock_impl.end_capture(c_python_object)
        return self.data

    def end_callback(self, data):
        import re
        log("data:")
        log(data)
        pattern = r"\[\s*?XPURT_PROF\s*?\]\s+(\S+?)\s+(\d+?)\s+(\S+?)\s+ns"
        matches = re.findall(pattern, data)
        log("match:")
        self.data = matches
        log(str(self.data))

class __GpuRuntimeProfiler:
    def __init__(self):
        pass
    def start_capture(self):
        import torch
        import torch.autograd.profiler
        self.prof = torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True, profile_memory=False).__enter__()
    def end_capture(self):
        self.prof.__exit__(None, None, None)
        # print(self.prof.table(row_limit=-1))
        table = self.prof.key_averages().table(row_limit=-1)
        ll = filter(lambda it:it.key.startswith('aten::'), self.prof.key_averages())
        ll = list(ll)
        data = []
        for it in list(ll):
            data.append((it.key, it.cuda_time_total_str.replace("us", "")))
            #print(it)
            #print(it.key)
            #print(it.self_cuda_time_total_str)
            #print(it.cuda_time_str)
        # print(type(self.prof.key_averages()))
        # print(self.prof.key_averages())
        self.data = data
        log(str(self.data))

RuntimeProfiler = __XpuRuntimeProfiler if True else __GpuRuntimeProfiler


class HookInstaller:
    def __init__(self, lib):
        self.lib = ctypes.CDLL(lib)
        c_python_object = ctypes.py_object(self)
        c_string_lib = ctypes.c_char_p(lib.encode('utf-8'))
        cuda_mock_impl.create_hook_installer(c_python_object, c_string_lib)
    def is_target_lib(self, name):
        print(f"HookInstaller is_taget_lib:{name}")
        assert False, "unimplement"

    def is_target_symbol(self, name):
        print(f"HookInstaller is_target_symbol:{name}")
        assert False, "unimplement"

    def new_symbol_name(self, name):
        return name


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


