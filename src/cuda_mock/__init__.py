from .cuda_mock_impl import *
from .dynamic_obj import *
from .triton_mock import *

import atexit

# check version
import sys, re
versions = re.findall(r"major=(\d+), minor=(\d+), micro=(\d+)", str(sys.version_info))[0]
int_version = (int(versions[0]) << 24) + (int(versions[1]) << 16) + (int(versions[2]) << 8)
# ignore lower 8 byte
if (get_build_version_int() >> 8) != (int_version >> 8):
    log(f"build with python version:{hex(get_build_version_int())} vs runtime python version:{hex(int_version)} mismatch", LOG_LEVEL.WARN)

atexit.register(uninitialize)

initialize()
