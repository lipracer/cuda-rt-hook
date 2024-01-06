#include <stddef.h>
#include <stdio.h>
#include <unistd.h>

#include <fstream>

#define EXPORT __attribute__((__visibility__("default")))
#define SUCCESS 0

extern "C" {

EXPORT int xpu_malloc(void** devPtr, uint64_t size, int kind) { 
    *devPtr = malloc(size);
    return SUCCESS; 
}

EXPORT int xpu_free(void* ptr) {
    free(ptr);
    return SUCCESS;
}

thread_local int gDeviceId = 0;

EXPORT int xpu_current_device(int* devId) {
    *devId = gDeviceId;
    return SUCCESS;
}

EXPORT int xpu_wait(void*) { 
    return SUCCESS; 
}

EXPORT int xpu_memcpy(void*, const void*, uint64_t, int) { return SUCCESS; }

EXPORT int xpu_set_device(int devId) {
    gDeviceId = devId;
    return SUCCESS;
}
}