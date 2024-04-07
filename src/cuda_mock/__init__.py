from .cuda_mock_impl import *
from .dynamic_obj import *
from .gpu_validation import *

import atexit

atexit.register(uninitialize)

initialize()
