import os
import cuda_mock

script_dir = os.path.dirname(os.path.abspath(__file__))
xpu_lib = cuda_mock.dynamic_obj(f'{script_dir}/xpu_libs.cxx').appen_compile_opts('-g -lpthread').compile().get_lib()
new_name = xpu_lib.split('/')
new_name[-1] = 'libxpurt.so'
new_name = '/'.join(new_name)
os.system(f'mv {xpu_lib} {new_name}')

cpp_code_0 = '''

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
    xpu_malloc(&devPtr, 16, 0);
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

EXPORT void test_xpu_api() {
    call_xpu_malloc();
    call_xpu_memcpy();
    call_xpu_free();

    call_xpu_malloc();
    call_xpu_memcpy();
    call_xpu_free();

    call_xpu_malloc();
    call_xpu_free();

    call_xpu_malloc();
    call_xpu_free();
    
    call_xpu_wait();
    call_xpu_wait();
    call_xpu_wait();
    call_xpu_set_device();
    call_xpu_set_device();

    call_xpu_current_device();
}

}

'''

cpp_code_1 = '''

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


EXPORT void test_xpu_api() {
    {
        void* ptr = nullptr;
        xpu_set_device(0);
        xpu_malloc(&ptr, 4, 0);
        xpu_free(ptr);
        xpu_wait(nullptr);
    }

    {
        void* ptr = nullptr;
        xpu_set_device(1);
        xpu_malloc(&ptr, 8, 1);
        xpu_free(ptr);
        xpu_wait(nullptr);
    }

    {
        void* ptr = nullptr;
        xpu_set_device(2);
        xpu_malloc(&ptr, 16, 0);
        xpu_free(ptr);
        xpu_wait(nullptr);
    }

    {
        void* ptr = nullptr;
        xpu_set_device(3);
        xpu_malloc(&ptr, 32, 1);
        xpu_free(ptr);
    }

}

}

'''

dummy_lib_0 = cuda_mock.dynamic_obj(cpp_code_0, True).appen_compile_opts('-g -lxpurt', '-L/tmp', '-Wl,-rpath,/tmp').compile().get_lib()
dummy_lib_1 = cuda_mock.dynamic_obj(cpp_code_1, True).appen_compile_opts('-g -lxpurt', '-L/tmp', '-Wl,-rpath,/tmp').compile().get_lib()

from cuda_mock import *
import ctypes

dummy_lib_0 = ctypes.CDLL(dummy_lib_0)
dummy_lib_1 = ctypes.CDLL(dummy_lib_1)


def test_hook_xpu_functions():
    cuda_mock.xpu_initialize()

    # first call
    dummy_lib_0.test_xpu_api()
    dummy_lib_1.test_xpu_api()
    

if __name__ == '__main__':
    test_hook_xpu_functions()