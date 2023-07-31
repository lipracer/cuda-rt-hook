// bar.cc
#include <stdio.h>

#ifdef STATIC_LIBRARY
#define EXPORT
#else
#define EXPORT __attribute__((__visibility__("default")))
#endif

EXPORT void* mmalloc(int) {
    printf("file:%s func:%s\n", __FILE__, __func__);
    return nullptr;
}

EXPORT void bar(void) {
    printf("file:%s func:%s\n", __FILE__, __func__);
}