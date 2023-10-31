## The plt hook technology used refers to [plthook](https://github.com/kubo/)  
#### mock pytorch cuda runtime interface

- update submodule
`git submodule update --init --recursive`

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

### 收集cuda 算子调用堆栈
- 找到nvcc安装路径
`which nvcc`  
- 用我们的nvcc替换系统的nvcc（我们只是在编译选项加了`-g`）  
`mv /usr/local/bin/nvcc /usr/local/bin/nvcc_b`  
`chmod 777 tools/nvcc`  
`cp tools/nvcc /usr/local/bin/nvcc`
- 构建并且安装pytorch
- 构建并且安装cuda_mock
- 注意要在import torch之后import cuda_mock
- 开始跑你的训练脚本
- 我们将会把堆栈打印到控制台

### example
`python test/test_import_mock.py`

### debug
- export LOG_LEVEL=0
