## The plt hook technology used refers to [plthook](https://github.com/kubo/)  
#### mock pytorch cuda runtime interface

- update submodule  
`git submodule update --init --recursive`

- build wheel package  
`python setup.py sdist bdist_wheel`

- direct install  
`pip install dist/*.whl`

### collect cuda operator call stack
1. find nvcc installed path
```bash
which nvcc
```
2. replace nvcc with my nvcc
```bash
mv /usr/local/bin/nvcc /usr/local/bin/nvcc_b
chmod 777 tools/nvcc
cp tools/nvcc /usr/local/bin/nvcc
```  
3. build and install pytorch
4. build and install cuda_mock
5. import cuda_mock after import torch
6. run your torch train script
7. we will dump the stack into console
