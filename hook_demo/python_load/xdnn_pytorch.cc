
#define EXPORT __attribute__((__visibility__("default")))

extern "C" {
EXPORT void xdnn_add();
EXPORT void xdnn_xblas_add();

EXPORT void xdnn_pytorch_add() { xdnn_add(); }

EXPORT void xdnn_pytorch_add_() { xdnn_xblas_add(); }
}