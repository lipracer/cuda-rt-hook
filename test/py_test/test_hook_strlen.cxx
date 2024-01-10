// test/test_hook_strlen.cnn

#include <stddef.h>
#include <stdio.h>
#include <unistd.h>

#include <fstream>
#include <iostream>

#include "logger/logger.h"

#define EXPORT __attribute__((__visibility__("default")))

extern "C" {

void* __origin_strlen = nullptr;

EXPORT size_t strlen(const char* str) {
    size_t len = (*reinterpret_cast<decltype(&strlen)>(__origin_strlen))(str);
    LOG(WARN) << "run into hook func str:" << str << "and len:" << len;
    return len;
}

std::ofstream ofs("/tmp/cuda_hook_file_logger_" + std::to_string(getpid()) + ".log"); 

EXPORT int __printf_chk(int flag, const char* fmt, va_list argp) {
    pid_t process_id = getpid();
    constexpr size_t kMax = 1024;
    char buf[kMax] = {"myprintf "};
    snprintf(buf + strlen(buf), kMax - strlen(buf), fmt, argp);
    ofs << buf;
    return 0;
}

}
