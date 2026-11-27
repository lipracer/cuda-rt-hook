import os
import subprocess
import sys

import pytest


def run_python(source, *args, trace=""):
    env = dict(os.environ, LOG_SYNC_MODE="1", LOG_LEVEL="WARN",
               HOOK_ENABLE_TRACE=trace)
    env.pop("TARGET_LIB_FILTER", None)
    result = subprocess.run(
        [sys.executable, "-c", source, *map(str, args)],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout
    return result.stdout


def test_python_backtrace_releases_frame_locals():
    output = run_python("""
import gc
import weakref
import cuda_mock

class Local:
    pass

def traced_call():
    local = Local()
    ref = weakref.ref(local)
    assert cuda_mock.backtrace_add_test(2, 3) == 5
    return ref

ref = traced_call()
gc.collect()
assert ref() is None
""")
    assert "Python frame:" in output
    assert "traced_call" in output


@pytest.fixture(scope="module")
def runtime_fixtures(tmp_path_factory):
    root = tmp_path_factory.mktemp("runtime-batch")
    provider = root / "provider.c"
    provider.write_text("""
#include <stddef.h>
static int device_status;
static int device_calls;
void set_device_status(int status) { device_status = status; }
int get_device_calls(void) { return device_calls; }

#ifndef OMIT_DEVICE
int cudaGetDevice(int *device) {
    *device = 3;
    ++device_calls;
    return device_status;
}
#endif
int cudaMalloc(void **ptr, size_t bytes) {
    if (bytes == 0) return 7;
    *ptr = (void *)0x1234;
    return 0;
}
int cudaFree(void *ptr) { return ptr ? 0 : 9; }
int batch_probe_one(void) { return 1; }
int batch_probe_two(void) { return 2; }
""")
    caller = root / "caller.c"
    caller.write_text("""
#include <stddef.h>
extern int cudaMalloc(void **, size_t);
extern int cudaFree(void *);
extern int batch_probe_one(void);
extern int batch_probe_two(void);
int allocate(void **ptr, size_t bytes) { return cudaMalloc(ptr, bytes); }
int release(void *ptr) { return cudaFree(ptr); }
int call_one(void) { return batch_probe_one(); }
int call_two(void) { return batch_probe_two(); }
""")
    extension = root / "extension.c"
    extension.write_text("""
int (*__origin_batch_probe_one)(void);
int (*__origin_batch_probe_two)(void);
int replace_one(void) { return __origin_batch_probe_one() + 10; }
int replace_two(void) { return __origin_batch_probe_two() + 20; }
""")

    def build(source, output, *flags):
        subprocess.run(
            ["cc", "-fPIC", "-shared", "-o", str(output), str(source), *flags],
            check=True,
        )

    build(extension, root / "libextension.so")
    for scenario, flags in (("with_device", ()),
                            ("without_device", ("-DOMIT_DEVICE",))):
        libs = root / scenario
        libs.mkdir()
        build(provider, libs / "libcudart.so.fixture", *flags)
        build(caller, libs / "libbatch_caller.so",
              f"-L{libs}", "-l:libcudart.so.fixture", "-Wl,-rpath,$ORIGIN",
              "-Wl,-z,now")
    return root


@pytest.mark.parametrize("scenario,expected_queries", [
    ("with_device", 4),
    ("without_device", 0),
])
def test_batch_hooks_and_optional_device_query(runtime_fixtures, scenario,
                                                expected_queries):
    output = run_python(r"""
import ctypes
import os
from pathlib import Path
import sys

root, scenario, expected_queries = sys.argv[1:]
root = Path(root)
libdir = root / scenario
provider = ctypes.CDLL(str(libdir / "libcudart.so.fixture"))
caller = ctypes.CDLL(str(libdir / "libbatch_caller.so"))
caller.allocate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
caller.release.argtypes = [ctypes.c_void_p]

# The only eligible caller imports allocation APIs, never cudaGetDevice.
os.environ["TARGET_LIB_FILTER"] = r"/libbatch_caller[.]so$"
import cuda_mock

assert (caller.call_one(), caller.call_two()) == (1, 2)
cuda_mock.xpu_initialize_with_hooks(
    r"/libbatch_caller[.]so$", root / "libextension.so",
    {"batch_probe_one": "replace_one", "batch_probe_two": "replace_two"},
)
assert (caller.call_one(), caller.call_two()) == (11, 22)

ptr = ctypes.c_void_p()
assert caller.allocate(ctypes.byref(ptr), 64) == 0 and ptr.value == 0x1234
assert caller.release(ptr) == 0
provider.set_device_status(42)
assert caller.allocate(ctypes.byref(ptr), 64) == 0 and ptr.value == 0x1234
assert caller.release(ptr) == 0
assert caller.allocate(ctypes.byref(ptr), 0) == 7
assert caller.release(None) == 9
assert provider.get_device_calls() == int(expected_queries)
""", runtime_fixtures, scenario, expected_queries,
        trace="cudaMalloc=2,cudaFree=2")
    assert "cudaMalloc:" in output
    assert "cudaFree:" in output
