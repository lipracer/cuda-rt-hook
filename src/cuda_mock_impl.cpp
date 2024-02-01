
#include <Python.h>

#include "cuda_mock.h"
#include "hook.h"
#include "logger/logger.h"
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

extern "C" {

HOOK_API int add(int lhs, int rhs) { return lhs + rhs; }

HOOK_API void initialize() { dh_initialize(); }

HOOK_API void uninitialize() { dh_uninitialize(); };

HOOK_API void internal_install_hook(HookString_t srcLib, HookString_t targetLib,
                                    HookString_t symbolName,
                                    HookString_t hookerLibPath,
                                    HookString_t hookerSymbolName) {
    dh_internal_install_hook(srcLib, targetLib, symbolName, hookerLibPath,
                             hookerSymbolName);
}

HOOK_API void internal_install_hook_regex(HookString_t srcLib,
                                          HookString_t targetLib,
                                          HookString_t symbolName,
                                          HookString_t hookerLibPath,
                                          HookString_t hookerSymbolName) {
    dh_internal_install_hook_regex(srcLib, targetLib, symbolName, hookerLibPath,
                                   hookerSymbolName);
}

HOOK_API void xpu_initialize() { xpu_dh_initialize(); }

HOOK_API void patch_runtime() { dh_patch_runtime(); }

#define CHECK_PYTHON_OBG()

HOOK_API bool call_python_method_bool(PyObject* py_instance, HookString_t name,
                                      HookString_t value) {
    Py_Initialize();
    CHECK(Py_IsInitialized(), "python interpreter uninitialized");

    PyGILState_STATE gstate = PyGILState_Ensure();
    // https://stackoverflow.com/questions/1796510/accessing-a-python-traceback-from-the-c-api
    // PyThreadState* tstate = PyThreadState_GET();

    CHECK(py_instance, "py_method:{0} py_instance empty!", name);

    PyObject* py_method = PyObject_GetAttrString(py_instance, name);
    CHECK(py_method, "py_method:{0} empty!", name);

    PyObject* py_value = PyTuple_Pack(1, PyUnicode_FromString(value));
    CHECK(py_value, "py_method:{0} py_value empty!", name);

    PyObject* py_result = PyObject_CallObject(py_method, py_value);
    CHECK(py_result, "py_method:{0} py_result empty!", name);

    auto native_result = PyObject_IsTrue(py_result);

    PyGILState_Release(gstate);
    return native_result;
}

HOOK_API HookString_t call_python_method_string(PyObject* py_instance,
                                                HookString_t name,
                                                HookString_t value) {
    Py_Initialize();
    CHECK(Py_IsInitialized(), "python interpreter uninitialized");

    PyGILState_STATE gstate = PyGILState_Ensure();
    // https://stackoverflow.com/questions/1796510/accessing-a-python-traceback-from-the-c-api
    // PyThreadState* tstate = PyThreadState_GET();

    CHECK(py_instance, "py_method:{0} py_instance empty!", name);

    PyObject* py_method = PyObject_GetAttrString(py_instance, name);
    CHECK(py_method, "py_method:{0} empty!", name);

    PyObject* py_value = PyTuple_Pack(1, PyUnicode_FromString(value));
    CHECK(py_value, "py_method:{0} py_value empty!", name);

    PyObject* py_result = PyObject_CallObject(py_method, py_value);
    CHECK(py_result, "py_method:{0} py_result empty!", name);

    PyObject* str_object = PyObject_Str(py_result);
    CHECK(str_object, "py_method:{0} PyObject_Str empty!", name);

    HookString_t native_string = PyUnicode_AsUTF8(str_object);
    CHECK(native_string, "py_method:{0} PyUnicode_AsUTF8 empty!", name);

    PyGILState_Release(gstate);
    return native_string;
}

HOOK_API void create_hook_installer(PyObject* py_instance, HookString_t lib) {
    auto is_target = [=](HookString_t name) {
        return call_python_method_bool(py_instance, "is_target_lib", name);
    };
    auto is_symbol = [=](HookString_t name) {
        return call_python_method_bool(py_instance, "is_target_symbol", name);
    };
    auto new_symbol = [=](HookString_t name) {
        return call_python_method_string(py_instance, "new_symbol_name", name);
    };
    dh_create_py_hook_installer(is_target, is_symbol, lib, new_symbol);
}
}
