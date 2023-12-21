#include <nanobind/nanobind.h>

#include "cuda_mock.h"
#include "xpu_mock.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(cuda_mock_impl, m) {
    m.def(
        "add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);
    m.def("initialize", []() { dh_initialize(); });
    m.def("internal_install_hook", [](const char* srcLib, const char* targetLib,
                                      const char* symbolName) {
        dh_internal_install_hook(srcLib, targetLib, symbolName);
    });
    m.def("internal_install_hook",
          [](const char* srcLib, const char* targetLib, const char* symbolName,
             const char* hookerLibPath, const char* hookerSymbolName) {
              dh_internal_install_hook(srcLib, targetLib, symbolName,
                                       hookerLibPath, hookerSymbolName);
          });
    m.def("xpu_initialize", []() { xpu_dh_initialize(); });
}
