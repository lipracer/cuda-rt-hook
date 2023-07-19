## The plt hook technology used refers to [plthook](https://github.com/kubo/)  
#### mock pytorch cuda runtime interface

- build wheel package  
`pip wheel .`

- direct install  
`pip install .`

### collect cuda operator call stack
- find nvcc installed path  
`which nvcc`  
- replace nvcc with my nvcc  
`mv /usr/local/bin/nvcc /usr/local/bin/nvcc_b`  
`chmod 777 tools/nvcc`  
`cp tools/nvcc /usr/local/bin/nvcc`
- build and install pytorch
- build and install cuda_mock
- import cuda_mock after import torch
- run your torch train script
- we will dump the stack into console

### debug
- export LOG_LEVEL=0
