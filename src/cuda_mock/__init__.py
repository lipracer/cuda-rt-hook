from .cuda_mock_impl import *
from .dynamic_obj import *
try:
    from .gpu_validation import *
except ImportError:
    print("GPU validation is not available. You must install PyTorch first.")

import atexit

# check version
import sys, re
versions = re.findall(r"major=(\d+), minor=(\d+), micro=(\d+)", str(sys.version_info))[0]
int_version = (int(versions[0]) << 24) + (int(versions[1]) << 16) + (int(versions[2]) << 8)
# ignore lower 8 byte
if (get_build_version_int() >> 8) != (int_version >> 8):
    print(f"warning: {hex(get_build_version_int())} vs {hex(int_version)} mismatch")

atexit.register(uninitialize)

initialize()
