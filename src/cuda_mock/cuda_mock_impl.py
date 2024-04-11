import os
import ctypes
from .dynamic_obj import *
import json
import re
import inspect
from .ctypes_helper import convert_arg_list_of_str

script_dir = os.path.dirname(os.path.abspath(__file__))
cuda_mock_impl = ctypes.CDLL(f'{script_dir}/libcuda_mock_impl.so')
cuda_mock_impl.print_hook_initialize.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p)]

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

def xpu_initialize():
    return cuda_mock_impl.xpu_initialize()

def print_hook_initialize(target_libs, target_symbols):
    target_libs = convert_arg_list_of_str(target_libs)
    target_symbols = convert_arg_list_of_str(target_symbols)
    cuda_mock_impl.print_hook_initialize(target_libs, target_symbols)
    

def patch_runtime():
    return cuda_mock_impl.patch_runtime()

def log(*args, level=0):
    caller_frame = inspect.currentframe().f_back
    caller_filename = inspect.getframeinfo(caller_frame).filename
    caller_lineno = inspect.getframeinfo(caller_frame).lineno
    new_args = [f'[{caller_filename}:{caller_lineno}]{arg}' for arg in args]
    new_args = [ctypes.c_char_p(arg.encode('utf-8')) for arg in new_args]
    if level == 0:
        return cuda_mock_impl.py_log_info(*new_args)
    elif level == 1:
        return cuda_mock_impl.py_log_warn(*new_args)
    else:
        return cuda_mock_impl.py_log_error(*new_args)

is_nvidia_gpu = os.environ.get('CUDA_VERSION', None) or os.environ.get('NVIDIA_VISIBLE_DEVICES', None)

class ProfileDataCollection:
    def __init__(self, device):
        self.data = []
        self.device = device
    def append_gpu(self, key, data):
        self.load_from_cache()
        self.append_implement(key, data, 'gpu')
        self.dump_to_cache()

    def append_xpu(self, key, data):
        self.load_from_cache()
        self.append_implement(key, data, 'xpu')
        self.dump_to_cache()

    def append_implement(self, key, data, device):
        data = self.statistic_data(data, device)
        for it in self.data:
            if key in it:
                it[key][device] = data
                return
        self.data.append({key : {device:data}})

    def get_xpu_time(self, data):
        return float(data[-1].strip().replace("ns", ""))

    def get_gpu_time(self, data):
        if data[0].startswith("aten::empty"):
            return 0.0
        if data[0] != 'total':
            return 0.0
        time_str = data[1].strip()
        if time_str.endswith("ns"):
            return float(time_str.replace("ns", ""))
        elif time_str.endswith("us"):
            return float(time_str.replace("us", "")) * 1000
        elif time_str.endswith("ms"):
            return float(time_str.replace("ms", "")) * 1000 * 1000
        elif time_str.endswith("s"):
            return float(time_str.replace("s", "")) * 1000 * 1000 * 1000
        raise RuntimeError(f"error uint:{time_str}")

    def statistic_data(self, data, device):
        total = 0.0
        for it in data:
            if is_nvidia_gpu:
                total += self.get_gpu_time(it)
            else:
                total += self.get_xpu_time(it)
        data = {"total" : total, "data" : data}
        return data

    def load_from_cache(self):
        pass
        # try:
        #     with open(f"{self.device}-profile_data.json", "rt") as f:
        #         self.data = json.load(f)
        # except Exception:
        #     pass

    def dump_to_cache(self):
        with open(f"{self.device}-profile_data.json", "wt") as f:
            json.dump(self.data, f, indent=4)

gProfileDataCollection = ProfileDataCollection("gpu" if is_nvidia_gpu else "xpu")
gDefaultTargetLib = ["libxpucuda.so", "libcuda.so"]
gDefaultTargetSymbols = ["__printf_chk", "printf","fprintf","__fprintf","vfprintf",]
class __XpuRuntimeProfiler:
    def __init__(self, target_libs = gDefaultTargetLib, target_symbols = gDefaultTargetSymbols):
        print_hook_initialize(target_libs=target_libs, target_symbols=target_symbols)
    def start_capture(self):
        cuda_mock_impl.print_hook_start_capture()

    def end_capture(self, op_key):
        log(f"op_key:{op_key}")
        c_python_object = ctypes.py_object(self)
        cuda_mock_impl.print_hook_end_capture(c_python_object)
        log(f"data:{self.data}")
        gProfileDataCollection.append_xpu(op_key, self.data)
        return self.data

    def end_callback(self, data):
        pattern = r"\[\s*?XPURT_PROF\s*?\]\s+(\S+?)\s+(\d+?)\s+(\S+?)\s+ns"
        matches = re.findall(pattern, data)
        self.data = matches


class __GpuRuntimeProfiler:
    def __init__(self, target_libs = gDefaultTargetLib, target_symbols = gDefaultTargetSymbols):
        print_hook_initialize(target_libs=target_libs, target_symbols=target_symbols)
    def start_capture(self):
        import torch
        from torch.profiler import profile, record_function, ProfilerActivity
        self.prof = profile(activities=[ProfilerActivity.CUDA], record_shapes=False).__enter__()
    def end_capture(self, op_key):
        self.prof.__exit__(None, None, None)
        # print(self.prof.table(row_limit=-1))
        table = self.prof.key_averages().table(sort_by="cuda_time_total", row_limit=-1)
        time_list = self.prof.key_averages()

        data = []
        for it in time_list:
            data.append((it.key, it.cuda_time_total_str))
            #print(it)
            #print(it.key)
            #print(it.self_cuda_time_total_str)
            #print(it.cuda_time_str)
        table = str(table)
        log(f"{table}")

        pattern = r"Self CUDA time total:(.*\s)"
        matches = re.findall(pattern, table)

        if len(matches) == 1:
            data.append(('total', matches[0]))
        else:
            data.append(('total', "0.0ms"))
        self.data = data
        gProfileDataCollection.append_gpu(op_key, self.data)
        log(f"{op_key}:{self.data}")

RuntimeProfiler = __GpuRuntimeProfiler if is_nvidia_gpu else __XpuRuntimeProfiler


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


