import os
import cuda_mock

cpp_code = '''

#include <stddef.h>
#include <stdio.h>
#include <unistd.h>

#include "logger/logger.h"

#include <fstream>
#include <iostream>

#define EXPORT __attribute__((__visibility__("default")))

extern "C" {

void * __origin_strlen = nullptr;
void * __origin_malloc = nullptr;

EXPORT void* my_malloc(size_t s) {
    puts("run into hook my_malloc");
    return reinterpret_cast<decltype(&my_malloc)>(__origin_malloc)(s);
}

EXPORT int new_strlen(const char* str) {
    puts("run into hook new_strlen");
    return reinterpret_cast<decltype(&strlen)>(__origin_strlen)(str);
}

}

'''

def test_hook_malloc():
    lib = cuda_mock.dynamic_obj(cpp_code, True).appen_compile_opts('-g').compile().get_lib()
    cuda_mock.internal_install_hook_regex(r"glibc\.so\..*", r"[^(glibc\.so)]", "strlen", str(lib), "new_strlen")


class PythonHookInstaller(cuda_mock.HookInstaller):
    def is_target_lib(self, name):
        print(name)
        return name.find("libcuda_mock_impl.so") != -1

    def is_target_symbol(self, name):
        # print(name)
        return name.find("strlen") != -1

    def new_symbol_name(self, name):
        return "new_" + name

def test_python_hook():
    lib = cuda_mock.dynamic_obj(cpp_code, True).appen_compile_opts('-g').compile().get_lib()
    installer = PythonHookInstaller(lib)

if __name__ == '__main__':
    test_hook_malloc()
    test_python_hook()