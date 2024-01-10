
#include "cuda_mock.h"
#include "xpu_mock.h"

// namespace nb = nanobind;
// using namespace nb::literals;

// NB_MODULE(cuda_mock_impl, m) {
//     m.def(
//         "add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);
//     m.def("initialize", []() { dh_initialize(); });
//     m.def("internal_install_hook", [](const char* srcLib, const char*
//     targetLib,
//                                       const char* symbolName) {
//         dh_internal_install_hook(srcLib, targetLib, symbolName);
//     });
//     m.def("internal_install_hook",
//           [](const char* srcLib, const char* targetLib, const char*
//           symbolName,
//              const char* hookerLibPath, const char* hookerSymbolName) {
//               dh_internal_install_hook(srcLib, targetLib, symbolName,
//                                        hookerLibPath, hookerSymbolName);
//           });
//     m.def("xpu_initialize", []() { xpu_dh_initialize(); });
// }

typedef const char* PHStr;

extern "C" {

HOOK_API int add(int lhs, int rhs) { return lhs + rhs; }

HOOK_API void initialize() { dh_initialize(); }

HOOK_API void uninitialize() { dh_uninitialize(); };

HOOK_API void internal_install_hook(PHStr srcLib, PHStr targetLib,
                                    PHStr symbolName, PHStr hookerLibPath,
                                    PHStr hookerSymbolName) {
    dh_internal_install_hook(srcLib, targetLib, symbolName, hookerLibPath,
                             hookerSymbolName);
}

HOOK_API void internal_install_hook_regex(PHStr srcLib, PHStr targetLib,
                                          PHStr symbolName, PHStr hookerLibPath,
                                          PHStr hookerSymbolName) {
    dh_internal_install_hook_regex(srcLib, targetLib, symbolName, hookerLibPath,
                                   hookerSymbolName);
}

HOOK_API void xpu_initialize() { xpu_dh_initialize(); }
}
