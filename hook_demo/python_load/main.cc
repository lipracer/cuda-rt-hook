#ifdef STATIC_LIBRARY
#define EXPORT extern
#else
#define EXPORT __attribute__((__visibility__("default")))
#endif

extern "C" {
EXPORT void xdnn_pytorch_add();

EXPORT void xdnn_xblas_add();
}

int main(int argc, char** argv) {
    xdnn_pytorch_add();

    xdnn_xblas_add();
    return 0;
}