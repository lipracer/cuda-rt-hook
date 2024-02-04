#define EXPORT __attribute__((__visibility__("default")))

#include <iostream>

extern "C" {
EXPORT void xdnn_add() {
    std::cout << __FILE__ << ":" << __func__ << std::endl;
}
}