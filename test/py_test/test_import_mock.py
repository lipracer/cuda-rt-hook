import os
import cuda_mock

cpp_code = '''

#include <stddef.h>
#include <stdio.h>
#include <unistd.h>

#include "logger/logger.h"

#include <fstream>

#define EXPORT __attribute__((__visibility__("default")))

extern "C" {

void * __origin_malloc = nullptr;

EXPORT void* my_malloc(size_t s) {
    LOG(WARN) << "run into hook";
    return reinterpret_cast<decltype(&my_malloc)>(__origin_malloc)(s);
}

}

'''

def test_hook_malloc():
    lib = cuda_mock.dynamic_obj(cpp_code, True).appen_compile_opts('-g').compile().get_lib()
    cuda_mock.internal_install_hook_regex(r"glibc\.so\..*", r"libcuda_mock_impl\.so", "malloc", str(lib), "my_malloc")


if __name__ == '__main__':
    test_hook_malloc()