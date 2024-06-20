## The plt hook technology used refers to [plthook](https://github.com/kubo/)  
#### mock pytorch cuda runtime interface

- update submodule  
`git submodule update --init --recursive`

- build wheel package  
`python setup.py sdist bdist_wheel`

- direct install  
`pip install dist/*.whl`

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
