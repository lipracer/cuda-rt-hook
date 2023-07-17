// bar.cc
#include <stdio.h>

#define EXPORT __attribute__((__visibility__("default")))

EXPORT void* mmalloc(int) {
    printf("%s\n", __FILE__);
    return nullptr;
}

EXPORT void bar(void) {
    printf("%s %s\n", __FILE__, __func__);
}