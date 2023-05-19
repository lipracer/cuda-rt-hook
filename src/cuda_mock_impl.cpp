#include <nanobind/nanobind.h>
#include "hook.h"

namespace nb = nanobind;

using namespace nb::literals;

NB_MODULE(cuda_mock_impl, m) {
    m.def("add", [](int a, int b) { return a + b; }, "a"_a, "b"_a);
    m.def("init", []()
          { cuda_mock::install_hook(); });
}
