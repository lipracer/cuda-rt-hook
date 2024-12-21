#include "hello.h"

#include <dlfcn.h>
#include <stdio.h>

// 导出的函数
void hello() {
    printf("byebye_PRELOAD\n");
    typedef void (*hello_func)();
    void* so_handle = dlopen("libhello.so", RTLD_NOW);
    if (!so_handle) {
        fprintf(stderr, "%s\n", dlerror());
        return;
    }
    hello_func orig_hello = (hello_func)dlsym(so_handle, "hello");
    if (!orig_hello) {
        fprintf(stderr, "%s\n", dlerror());
        dlclose(so_handle);
        return;
    }
    printf("calling original hello(): ");
    orig_hello();
}

// void hello() { printf("byebye_PRELOAD\n"); }