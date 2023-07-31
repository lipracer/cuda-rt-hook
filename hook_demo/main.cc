#ifdef STATIC_LIBRARY
#define EXPORT extern
#else
#define EXPORT __attribute__((__visibility__("default")))
#endif

EXPORT void* mmalloc(int);

int main(int argc, char** argv) {
    mmalloc(0);
    return 0;
}