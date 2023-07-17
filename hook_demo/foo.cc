// foo.cc
#include <stdio.h>

#define EXPORT __attribute__((__visibility__("default")))

EXPORT void bar();

EXPORT void* mmalloc(int) {
    (void)bar();
    (void)bar();
    printf("%s\n", __FILE__);
    return nullptr;
}