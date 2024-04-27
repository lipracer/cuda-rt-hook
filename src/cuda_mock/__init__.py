from .cuda_mock_impl import *
from .dynamic_obj import *
try:
    from .gpu_validation import *
except ImportError:
    print("GPU validation is not available. You must install PyTorch first.")

import atexit

atexit.register(uninitialize)

initialize()
