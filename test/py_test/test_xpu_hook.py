import os
import cuda_mock

script_dir = os.path.dirname(os.path.abspath(__file__))
xpu_lib = cuda_mock.DynamicObj(f'{script_dir}/xpu_libs.cxx').appen_compile_opts('-g -lpthread').compile().get_lib()
new_name = xpu_lib.split('/')
new_name[-1] = 'libxpurt.so'
new_name = '/'.join(new_name)
os.system(f'mv {xpu_lib} {new_name}')

cpp_code = '''

#include <stddef.h>
#include <stdio.h>
#include <unistd.h>

#include <fstream>

#define EXPORT __attribute__((__visibility__("default")))

extern "C" {

EXPORT int xpu_malloc(void**, uint64_t, int);
EXPORT int xpu_free(void*);
EXPORT int xpu_current_device(int*);
EXPORT int xpu_wait(void*);
EXPORT int xpu_memcpy(void*, const void*, uint64_t, int);
EXPORT int xpu_set_device(int devId);

void* devPtr = nullptr;

EXPORT void call_xpu_malloc() {
    xpu_malloc(&devPtr, 0, 0);
}
EXPORT void call_xpu_free() {
    xpu_free(devPtr);
}
EXPORT void call_xpu_current_device(){
    int devId = 0;
    xpu_current_device(&devId);
}
EXPORT void call_xpu_wait() {
    xpu_wait(nullptr);
}
EXPORT void call_xpu_memcpy() {
    xpu_memcpy(devPtr, (char*)devPtr + 1, 1, 1);
}
EXPORT void call_xpu_set_device(){
    xpu_set_device(0);
}

}

'''

dummy_lib = cuda_mock.DynamicObj(cpp_code, True).appen_compile_opts('-g -lxpurt', '-L/tmp', '-Wl,-rpath,/tmp').compile().get_lib()
from cuda_mock import *
import ctypes

dummy_lib = ctypes.CDLL(dummy_lib)


def test_hook_xpu_functions():
    cuda_mock.xpu_initialize()

    # first call
    dummy_lib.call_xpu_current_device()
    
    dummy_lib.call_xpu_malloc()
    dummy_lib.call_xpu_memcpy()
    dummy_lib.call_xpu_free()

    dummy_lib.call_xpu_malloc()
    dummy_lib.call_xpu_memcpy()
    dummy_lib.call_xpu_free()

    dummy_lib.call_xpu_malloc()
    dummy_lib.call_xpu_free()

    dummy_lib.call_xpu_malloc()
    dummy_lib.call_xpu_free()
    
    dummy_lib.call_xpu_wait()
    dummy_lib.call_xpu_wait()
    dummy_lib.call_xpu_wait()
    dummy_lib.call_xpu_set_device()
    dummy_lib.call_xpu_set_device()

if __name__ == '__main__':
    test_hook_xpu_functions()