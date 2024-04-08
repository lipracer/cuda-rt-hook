import sys, os
import time
import socket
import torch

def validation_log_info(*args):
    print(*args)

def validation_log_debug(*args):
    pass
    # print(*args)

class gpu_validation:
    global_op_index = 0

    def __init__(self, model_key, atol, rtol, address, port, cache_dir='/tmp'):
        self.model_key = model_key
        self.atol = atol
        self.rtol = rtol

        self.address = address
        self.port = port

        self.cache_dir = cache_dir

    def send_tensor(self, tensor):
        self.send_tensor_impl(tensor)
    
    def recv_tensor(self):
        return self.recv_tensor_impl()

    # lhs is golden tensor
    def tensor_allclose(self, lhs, rhs):
        return str(torch.allclose(lhs, rhs, self.atol, self.rtol))

    def barrier(self):
        assert False, "not implement"

    def validate(self, tensor):
        assert False, "not implement"

    def get_op_index(self):
        index = gpu_validation.global_op_index
        return index

    def increase_index(self):
        gpu_validation.global_op_index += 1

    def get_tensor_file_name(self):
        return self.model_key + "_" + str(self.port) + "_" + str(self.get_op_index())

    def get_tensor_file_path(self):
        return os.path.join(self.cache_dir, self.get_tensor_file_name())


class gpu_validation_master(gpu_validation):
    def __init__(self, model_key, atol, rtol, address, port, cache_dir):
        super().__init__(model_key, atol, rtol, address, port, cache_dir)

    def validate(self, tensor):
        self.send_tensor(tensor)
        self.result = self.recv_result()

    def recv_result(self):
        assert False, "not implement"

class gpu_validation_client(gpu_validation):
    def __init__(self, model_key, atol, rtol, address, port, cache_dir):
        super().__init__(model_key, atol, rtol, address, port, cache_dir)

    def validate(self, tensor):
        validation_tensor = self.recv_tensor()
        result = self.tensor_allclose(tensor, validation_tensor)
        self.send_result(result)

    def send_result(self):
        assert False, "not implement"

socket_recv_file_length_buffer_size = 8
socket_recv_buffer_size = 1024 * 1024

def get_file_length(file):
    file.seek(0, 2)
    file_length = file.tell()
    file.seek(0)
    return file_length

class socket_gpu_validation_master(gpu_validation_master):
    def __init__(self, model_key, atol, rtol, address, port, cache_dir='/tmp'):
        super().__init__(model_key, atol, rtol, address, port, cache_dir)
        self.get_socket()

    def send_tensor_impl(self, tensor):
        torch.save(tensor, self.get_tensor_file_path())
        with open(self.get_tensor_file_path(), 'rb') as file:
            file_length = get_file_length(file)
            length_bytes = file_length.to_bytes(socket_recv_file_length_buffer_size, byteorder='big')
            validation_log_debug(f"file length size:{len(length_bytes)}")
            self.connection.sendall(length_bytes)

            while True:
                read_length = file_length if file_length < socket_recv_buffer_size else socket_recv_buffer_size
                binary_data = file.read(read_length)
                validation_log_debug(f"send all data size:{len(binary_data)}")
                self.connection.sendall(binary_data)
                file_length -= read_length
                if file_length == 0:
                    break

    def recv_result(self):
        data = self.connection.recv(socket_recv_buffer_size)
        return data.decode('utf-8')
    
    def get_socket(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        server_address = (self.address, self.port)
        validation_log_info('server address {} port {}'.format(*server_address))
        server_socket.bind(server_address)

        server_socket.listen(1)

        validation_log_info('waiting connect...')
        connection, client_address = server_socket.accept()
        validation_log_info(f'get client connect address:{client_address}')
        self.connection = connection
            
    def __del__(self):
        if hasattr(self, "connection"):
            self.connection.close()
        else:
            validation_log_info("has not connect")

class socket_gpu_validation_client(gpu_validation_client):
    def __init__(self, model_key, atol, rtol, address, port, cache_dir='/tmp'):
        super().__init__(model_key, atol, rtol, address, port, cache_dir)
        self.connect()

    def recv_tensor_impl(self):
        byte_data = self.connection.recv(socket_recv_file_length_buffer_size)
        file_length = int.from_bytes(byte_data, byteorder='big')
        validation_log_debug(f"validate tensor file size:{file_length}")
        recv_length = 0
        with open(self.get_tensor_file_path(), 'wb') as file:
            while True:
                byte_data = self.connection.recv(socket_recv_buffer_size)
                file.write(byte_data)
                recv_length += len(byte_data)
                validation_log_debug(f"current recv data size:{recv_length}")
                if recv_length == file_length:
                    break

        return torch.load(self.get_tensor_file_path())

    def send_result(self, result):
        binary_data = result.encode('utf-8')
        self.connection.sendall(binary_data)


    def connect(self):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (self.address, self.port)
        validation_log_info("connect {} {}".format(*server_address))
        client_socket.connect(server_address)
        self.connection = client_socket

    def __del__(self):
        if hasattr(self, "connection"):
            self.connection.close()
        else:
            validation_log_info("has not connect")

def exec_shell(cmd):
    p = os.popen(cmd)
    result = p.read().strip()
    validation_log_debug(f"exec {cmd} result:{result}")
    return_code = p.close()
    if return_code is None:
        return 0
    validation_log_info(f"exec {cmd} failed with result:{result}")
    return -1

def sync_pull_file(address, name, cache_dir):
    def find_file():
        for file_name in file_list:
            if file_name.endswith(name):
                return True
        return False

    while True: 
        file_list = os.popen(f"bcecmd bos ls -a {address}").read().strip()
        file_list = file_list.split('\n')
        if find_file():
            break
        time.sleep(0.001)
    bos_file_path = os.path.join(address, name)
    exec_shell(f"bcecmd bos cp {bos_file_path} {cache_dir}/ -y")
    validation_log_debug(f"pull file:{os.path.join(cache_dir, name)}")
    return os.path.join(cache_dir, name)

class bos_gpu_validation_master(gpu_validation_master):
    def __init__(self, model_key, atol, rtol, address, port, cache_dir='/tmp'):
        super().__init__(model_key, atol, rtol, address, port, cache_dir)
        if not self.address.endswith('/'):
            self.address += '/'


    def send_tensor_impl(self, tensor):
        validation_log_info(self.get_tensor_file_path())
        torch.save(tensor, self.get_tensor_file_path())

        try_count = 3
        while try_count > 0:
            if exec_shell(f"bcecmd bos cp {self.get_tensor_file_path()} {self.address} -y") == 0:
                break
            try_count -= 1
    
    def recv_result(self):
        result = sync_pull_file(self.address, self.get_tensor_file_name()+ "_result", self.cache_dir)
        with open(result, 'rt+') as f:
            result = f.read()
            return result


class bos_gpu_validation_client(gpu_validation_client):

    def __init__(self, model_key, atol, rtol, address, port, cache_dir='/tmp'):
        super().__init__(model_key, atol, rtol, address, port, cache_dir)
        if not self.address.endswith('/'):
            self.address += '/'

    def recv_tensor_impl(self):
        tensor_path = sync_pull_file(self.address, self.get_tensor_file_name(), self.cache_dir)
        return torch.load(tensor_path)

    def send_result(self, result):
        result_file = self.get_tensor_file_path()+ "_result"
        validation_log_info("send result:", result_file)
        with open(result_file, 'wt+') as f:
            f.write(result)
        
        exec_shell(f"bcecmd bos cp {result_file} { self.address} -y")

token = "test6"
def mytest_bos_master():
    master = bos_gpu_validation_master(token, 1e-3, 1e-3, "", 1000)

    lhs = torch.ones(3, 4)
    rhs = torch.ones(3, 4)
    ret = lhs + rhs
    master.validate(ret)
    master.increase_index()



def mytest_bos_client():
    client = bos_gpu_validation_client(token, 1e-3, 1e-3, "", 1000)

    lhs = torch.ones(3, 4)
    rhs = torch.ones(3, 4)
    ret = lhs + rhs
    client.validate(ret)
    client.increase_index()

SOCKET_PORT = 1002
TEST_SHAPE = (1024, 2, 4)

def mytest_socket_master():
    validation = socket_gpu_validation_master(token, 1e-3, 1e-3, "127.0.0.1", SOCKET_PORT)

    lhs = torch.ones(*TEST_SHAPE)
    rhs = torch.ones(*TEST_SHAPE)
    ret = lhs + rhs
    validation.validate(ret)
    validation.increase_index()

    ret = lhs - rhs
    validation.validate(ret)
    validation.increase_index()

def mytest_socket_client():
    validation = socket_gpu_validation_client(token, 1e-3, 1e-3, "127.0.0.1", SOCKET_PORT, "./")

    lhs = torch.ones(*TEST_SHAPE)
    rhs = torch.ones(*TEST_SHAPE)
    ret = lhs + rhs
    validation.validate(ret)
    validation.increase_index()

    ret = lhs - rhs
    validation.validate(ret)
    validation.increase_index()

import torch
from torch.utils._pytree import tree_flatten, tree_map
import functools
from torch.utils._python_dispatch import TorchDispatchMode

class GpuValidation(TorchDispatchMode):
    '''
    master is the device will be validation
    client is golden device 
    '''
    def __init__(self, is_gpu, model_key, atol, rtol, address, port=SOCKET_PORT, cache_dir='/tmp', use_bos=False):
        self.is_gpu = is_gpu
        if not use_bos:
            master = socket_gpu_validation_master
            client = socket_gpu_validation_client
        else:
            master = bos_gpu_validation_master
            client = bos_gpu_validation_client
        if is_gpu:
            self.validation = client(model_key, atol, rtol, address, port, cache_dir)
            validation_log_debug(f"initialize client validation completed!")
        else:
            self.validation = master(model_key, atol, rtol, address, port, cache_dir)
            validation_log_debug(f"initialize master validation completed!")

    def __torch_dispatch__(self, op, types, dev_args=(), dev_kwargs=None):
        validation_log_debug(f"will call op:{op}")
        result = op(*dev_args, **dev_kwargs)
        validation_log_debug(f"{op} result type: {type(result)}")
        if isinstance(result, torch.Tensor):
            validation_log_debug(f"will validate op:{op}")
            self.validation.validate(result)
            self.validation.increase_index()

            if not self.is_gpu:
                validation_log_info(f"validate op:{op} result:{self.validation.result}")
        else:
            validation_log_debug(f'skip not tensor result:{type(result)} op:{op}')

        return result


import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层的全连接层
        self.relu = nn.ReLU()  # 非线性激活函数
        self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层的全连接层

    def forward(self, x):
        x = self.fc1(x)  # 输入层到隐藏层的线性变换
        x = self.relu(x)  # 非线性激活函数
        x = self.fc2(x)  # 隐藏层到输出层的线性变换
        return x

def mytest_validation_master():
    with GpuValidation(False, token, 1e-3, 1e-3, "127.0.0.1", SOCKET_PORT, './'):
        input_size = 5
        hidden_size = 10
        output_size = 2
        model = SimpleNN(input_size, hidden_size, output_size)

        # 准备输入数据
        input_data = torch.randn(3, input_size)  # 创建一个3x5的随机输入张量

        # 将输入数据传递给模型，获取模型的输出
        output = model(input_data)

        # 打印模型输出
        print("模型输出：", output)

def mytest_validation_client():
    with GpuValidation(True, token, 1e-3, 1e-3, "127.0.0.1", SOCKET_PORT):
        input_size = 5
        hidden_size = 10
        output_size = 2
        model = SimpleNN(input_size, hidden_size, output_size)

        # 准备输入数据
        input_data = torch.randn(3, input_size)  # 创建一个3x5的随机输入张量

        # 将输入数据传递给模型，获取模型的输出
        output = model(input_data)

        # 打印模型输出
        print("模型输出：", output)

if __name__ == "__main__":
    if sys.argv[1] == "1":
        validation_log_info("client")
        mytest_validation_client()
    else:
        validation_log_info("master")
        mytest_validation_master()
