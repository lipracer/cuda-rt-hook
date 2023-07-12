#### mock pytorch cuda runtime interface

- build wheel package  
`pip wheel .`

- direct install  
`pip install .`

### collect cuda operator call stack
- find nvcc installed path  
`where nvcc`  
- replace nvcc with my nvcc  
`mv /usr/local/bin/nvcc /usr/local/bin/nvcc_bc`  
`chmod 777 tools/nvcc`  
`cp tools/nvcc /usr/local/bin/nvcc`
- build and install pytorch
- build and install cuda_mock
- import cuda_mock after import torch
- run your torch train script
- we will dump the stack into backtrace.log and save it to current path
- use parser script in content of tools parse backtrace.log

### debug
- export LOG_LEVEL=0