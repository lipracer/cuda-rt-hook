from .cuda_mock_impl import *
from .dynamic_obj import *
import atexit

atexit.register(uninitialize)

initialize()
