// foo.cc
#include <stdio.h>
#ifdef STATIC_LIBRARY
#define EXPORT
#else
#define EXPORT __attribute__((__visibility__("default")))
#endif

#ifdef STATIC_LIBRARY
EXPORT void bar() { printf("file:%s func:%s\n", __FILE__, __func__); }
#else
EXPORT void bar();
#endif

EXPORT void* mmalloc(int) {
    (void)bar();
    (void)bar();
    printf("file:%s func:%s\n", __FILE__, __func__);
    return nullptr;
}