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

### 收集统计xpu runtime 内存分配信息/`xpu_wait`调用堆栈
- 打印`xpu_malloc`调用序列，统计实时内存使用情况以及历史使用的峰值内存，排查内存碎片问题
- 打印`xpu_wait`调用堆栈，排查流水中断处问题
- 注意要在`import torch`/`import paddle`之后`import cuda_mock; cuda_mock.xpu_initialize()`
- 使用方法:

    ```python
    import paddle
    import cuda_mock; cuda_mock.xpu_initialize() # 加入这一行
    ```

### 实现自定义hook函数  
- 实现自定义hook installer例子:
    ```python
    class PythonHookInstaller(cuda_mock.HookInstaller):
        def is_target_lib(self, name):
            return name.find("libcuda_mock_impl.so") != -1
        def is_target_symbol(self, name):
            return name.find("malloc") != -1
    lib = cuda_mock.dynamic_obj(cpp_code, True).appen_compile_opts('-g').compile().get_lib()
    installer = PythonHookInstaller(lib)
    ```

- 实现hook回调接口 `PythonHookInstaller`  
- 构造函数需要传入自定义hook函数的库路径（绝对路径 并且 传入库中必须存在与要替换的函数名字以及类型一致的函数 在hook发生过程中，将会把原函数的地址写入以`__origin_`为开头目标`symbol`接口的变量中，方便用户拿到原始函数地址 参考:`test/py_test/test_import_mock.py:15`处定义）
- `is_target_lib` 是否是要hook的目标函数被调用的library
- `is_target_symbol` 是否是要hook的目标函数名字（上面接口返回True才回调到这个接口）
- `new_symbol_name` 构造函数中传入共享库中的新的用于替换的函数名字，参数`name`：当前准备替换的函数名字
- `dynamic_obj` 可以运行时编译c++ code，支持引用所有模块：`logger`、`statistics`



### example  
- ```python test/test_import_mock.py```

### debug
- ```export LOG_LEVEL=WARN,TRACE=INFO```

### 环境变量

| 环境变量 | 用法示例 | 可选值 | 默认值 | 说明 |
| ------ | ------- | ----- | ----- | ---- |
| LOG_LEVEL | `export LOG_LEVEL=WARN,TRACE=INFO` | 日志级别有:INFO,WARN,ERROR,FATAL, 日志模块有: PROFILE,TRACE,HOOK,PYTHON,LAST | 全局日志级别默认为WARN,各个日志模块的默认日志级别为INFO | 日志级别, 日志模块级别 |
| HOOK_ENABLE_TRACE | `export HOOK_ENABLE_TRACE='xpuMemcpy=1,xpuSetDevice=0'`  | xpuMalloc,xpuFree,xpuWait,xpuMemcpy,xpuSetDevice,xpuCurrentDeviceId | 默认所有接口的的值均为0,即所有接口默认关闭backtrace | 是否开启backtrace |
| LOG_OUTPUT_PATH |  `export LOG_OUTPUT_PATH='/tmp/'` |  日志输出文件夹 | - | 是否将日志重定向到文件 |
