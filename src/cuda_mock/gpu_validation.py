from cuda_mock.cuda_mock_impl import *
import sys, os
import time
import socket
import subprocess
import torch

def validation_log_info(*args):
    log(*args, 0)

def validation_log_debug(*args):
    log(*args, 0)

def validation_log_warn(*args):
    log(*args, 1)

def subprocess_log(*args):
    msg = ''
    for arg in args:
        msg += str(arg)
    print(f"[{os.getpid()}]{msg}")

class gpu_validation:
    global_op_index = 0

    def __init__(self, model_key, atol, rtol, address, port, cache_dir='/tmp', fallback=False):
        self.model_key = model_key
        self.atol = atol
        self.rtol = rtol

        self.address = address
        self.port = port

        self.cache_dir = cache_dir
        self.fallback = fallback

    def send_tensor(self, tensor):
        self.send_tensor_impl(tensor)
    
    def recv_tensor(self):
        return self.recv_tensor_impl()

    def send_msg(self, msg):
        self.send_msg_impl(msg)
    
    def recv_msg(self):
        return self.recv_msg_impl()

    # lhs is golden tensor
    def tensor_allclose(self, lhs, rhs, lhs_name, rhs_name):
        error_str = ''
        assert lhs_name == rhs_name,f"op_name mismatch {lhs_name} vs {rhs_name}"
        if lhs.shape != rhs.shape:
            error_str = f"shape mismatch {lhs.shape} vs {rhs.shape}"
        elif lhs.dtype != rhs.dtype:
            error_str = f"dtype mismatch {lhs.dtype} vs {rhs.dtype}"
        else:
            error_str = str(torch.allclose(lhs, rhs, self.atol, self.rtol))
        self.result = error_str
        return error_str

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

    def get_golden_tensor_file_name(self):
        return self.model_key + "_" + str(self.port) + "_" + str(self.get_op_index()) + "_golden"

    def get_tensor_file_path(self):
        return os.path.join(self.cache_dir, self.get_tensor_file_name())


class gpu_validation_master(gpu_validation):

    def validate(self, tensor, op_name):
        self.send_msg(op_name)
        self.send_tensor(tensor)
        self.result = self.recv_result_impl()

        if self.fallback:
            validation_log_debug("recv fallback tensor!")
            self.golden_result = self.recv_tensor()

class gpu_validation_client(gpu_validation):

    def validate(self, tensor, op_name):
        name = self.recv_msg()
        validation_tensor = self.recv_tensor()
        result = self.tensor_allclose(tensor, validation_tensor, op_name, name)
        self.send_result_impl(result)

        if self.fallback:
            validation_log_debug("send fallback tensor!")
            self.send_tensor(tensor)


socket_recv_file_length_buffer_size = 8
socket_recv_buffer_size = 1024 * 1024

def get_file_length(file):
    file.seek(0, 2)
    file_length = file.tell()
    file.seek(0)
    return file_length

def recv_msg(socket):
    byte_data = socket.recv(socket_recv_file_length_buffer_size)
    msg_length = int.from_bytes(byte_data, byteorder='big')
    byte_data = socket.recv(msg_length)
    return byte_data.decode('utf-8')

def send_msg(socket, msg):
    bytedata = msg.encode('utf-8')
    length = len(bytedata)
    blength = length.to_bytes(socket_recv_file_length_buffer_size, byteorder='big')
    socket.sendall(blength)
    socket.sendall(bytedata)

def send_tensor(obj, tensor):
    torch.save(tensor, obj.get_tensor_file_path())
    with open(obj.get_tensor_file_path(), 'rb') as file:
        file_length = get_file_length(file)
        length_bytes = file_length.to_bytes(socket_recv_file_length_buffer_size, byteorder='big')
        obj.connection.sendall(length_bytes)
        while True:
            read_length = file_length if file_length < socket_recv_buffer_size else socket_recv_buffer_size
            binary_data = file.read(read_length)
            validation_log_debug(f"send all data size:{len(binary_data)}")
            obj.connection.sendall(binary_data)
            file_length -= read_length
            if file_length == 0:
                break

def recv_tensor(obj):
    byte_data = obj.connection.recv(socket_recv_file_length_buffer_size)
    file_length = int.from_bytes(byte_data, byteorder='big')
    validation_log_debug(f"validate tensor file size:{file_length}")
    recv_length = 0
    with open(obj.get_tensor_file_path(), 'wb') as file:
        while True:
            byte_data = obj.connection.recv(socket_recv_buffer_size)
            file.write(byte_data)
            recv_length += len(byte_data)
            validation_log_debug(f"current recv data size:{recv_length}")
            if recv_length == file_length:
                break
    return torch.load(obj.get_tensor_file_path())


class socket_gpu_validation_master(gpu_validation_master):
    def __init__(self, model_key, atol, rtol, address, port, cache_dir='/tmp', fallback=False):
        super().__init__(model_key, atol, rtol, address, port, cache_dir, fallback)
        self.get_socket()

    def send_msg_impl(self, msg):
        send_msg(self.connection, msg)

    def recv_msg_impl(self):
        return recv_msg(self.connection)

    def send_tensor_impl(self, tensor):
        send_tensor(self, tensor)

    def recv_tensor_impl(self):
        return recv_tensor(self)

    def recv_result_impl(self):
        return self.recv_msg_impl()

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
    def __init__(self, model_key, atol, rtol, address, port, cache_dir='/tmp', fallback=False):
        super().__init__(model_key, atol, rtol, address, port, cache_dir, fallback)
        self.connect()

    def send_msg_impl(self, msg):
        return send_msg(self.connection, msg)

    def recv_msg_impl(self):
        return recv_msg(self.connection)

    def send_tensor_impl(self, tensor):
        send_tensor(self, tensor)
                
    def recv_tensor_impl(self):
        return recv_tensor(self)

    def send_result_impl(self, result):
        return self.send_msg_impl(result)

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

def exec_shell(command):
    subprocess_log(f"exec command:{command}")
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode
    except subprocess.CalledProcessError as e:
        subprocess_log("Error:", e)
        return -1
    return -1

def sync_pull_file(address, name, cache_dir, fuzzy_matching=False):
    new_file_name = ''
    def find_file():
        nonlocal new_file_name
        for file_name in file_list:
            if fuzzy_matching:
                new_file_name = file_name.split(" ")[-1]
                if new_file_name.startswith(name):
                    return True
            elif file_name.endswith(name):
                new_file_name = name
                return True
        return False

    while True: 
        file_list = os.popen(f"bcecmd bos ls -a {address}").read().strip()
        file_list = file_list.split('\n')
        if find_file():
            break
        time.sleep(0.5)
    bos_file_path = os.path.join(address, new_file_name)

    try_count = 3
    while try_count > 0:
        if exec_shell(f"bcecmd bos cp {bos_file_path} {cache_dir}/ -y") == 0:
            break
        try_count -= 1
    assert try_count >= 0, "bos_send_file send {file} fail!"

    
    validation_log_debug(f"pull file:{os.path.join(cache_dir, new_file_name)}")

    return os.path.join(cache_dir, new_file_name)

global_bos_msg_index = 0

def encode_path_msg(name, msg):
    global global_bos_msg_index
    result = f"{name}_{global_bos_msg_index}_msg_{msg}"
    global_bos_msg_index += 1
    return result

import multiprocessing
from multiprocessing import Queue
import time
from datetime import datetime
import shutil
from functools import partial

class TaskContext:
    def __init__(self):
        self.queue = Queue()
        self.p = multiprocessing.Process(
            target=self.task_consumer,
            args=(),
            kwargs={},
        )
        self.p.start()
    
    def task_consumer(self):
        subprocess_log(f"start background process task count:{self.queue.qsize()}")
        while True:
            if self.queue.empty():
                time.sleep(0.01)
                continue
            subprocess_log(f"remain task count:{self.queue.qsize()}")
            task = self.queue.get()
            if isinstance(task, TerminateTask):
                break
            task()

    def put_task(self, task):
        self.queue.put(task)
    
    def task_over(self):
        validation_log_info("task over wait sync file!")
        self.queue.put(TerminateTask())
        self.p.join()

def uninitialize():
    gTaskContext.task_over()

import atexit
atexit.register(uninitialize)

class SimpleTask:
    def __init__(self, file, dst_path):
        self.file = file
        self.dst_path = dst_path

    def __call__(self):
        try_count = 3
        while try_count > 0:
            if exec_shell(f"bcecmd bos cp {self.file} {self.dst_path} -y") == 0:
                exec_shell(f"rm -f {self.file}")
                break
            try_count -= 1
        assert try_count >= 0, "bos_send_file send {file} fail!"

class TerminateTask:
    pass


gTaskContext = TaskContext()

def bos_send_file_impl(obj, file):
    try_count = 3
    while try_count > 0:
        if exec_shell(f"bcecmd bos cp {file} {obj.address} -y") == 0:
            break
        try_count -= 1
    assert try_count >= 0, "bos_send_file send {file} fail!"
        

def bos_send_file(obj, file, sync=True):
    if sync:
        bos_send_file_impl(obj, file)
    else:
        gTaskContext.put_task(SimpleTask(file, obj.address))


def bos_send_msg(obj, msg, sync=True):
    msg_path = encode_path_msg(obj.get_tensor_file_path(), msg)
    with open(msg_path, "wt+") as f:
        f.write(msg)
    bos_send_file(obj, msg_path, sync)

def bos_recv_msg(obj):
    global global_bos_msg_index
    op_name_path = sync_pull_file(obj.address, f"{obj.get_tensor_file_name()}_{global_bos_msg_index}_msg_", obj.cache_dir, True)
    global_bos_msg_index += 1
    op_name = ''
    with open(op_name_path, "rt+") as f:
        op_name = f.read().strip()
    bos_name = op_name_path.split('/')[-1]
    # assert exec_shell(f"bcecmd bos rm {os.path.join(obj.address, bos_name)} -y")==0, "clear msg fail!"
    return op_name


class bos_gpu_validation_master(gpu_validation_master):
    def __init__(self, model_key, atol, rtol, address, port, cache_dir='/tmp', fallback=False, offline=False):
        super().__init__(model_key, atol, rtol, address, port, cache_dir, fallback)
        if not self.address.endswith('/'):
            self.address += '/'
        self.offline = offline

    def recv_msg_impl(self):
        return bos_recv_msg(self)

    def send_msg_impl(self, msg):
        sync = False if self.offline else True
        bos_send_msg(self, msg, sync)

    def send_tensor_impl(self, tensor):
        validation_log_info(self.get_tensor_file_path())
        torch.save(tensor, self.get_tensor_file_path())
        sync = False if self.offline else True
        bos_send_file(self, self.get_tensor_file_path(), sync)

    def recv_tensor_impl(self):
        tensor_path = sync_pull_file(self.address, self.get_tensor_file_name(), self.cache_dir)
        return torch.load(tensor_path)

    def recv_result_impl(self):
        if self.offline:
            return ""
        return self.recv_msg_impl()

class bos_gpu_validation_client(gpu_validation_client):

    def __init__(self, model_key, atol, rtol, address, port, cache_dir='/tmp', fallback=False, offline=False):
        super().__init__(model_key, atol, rtol, address, port, cache_dir, fallback)
        if not self.address.endswith('/'):
            self.address += '/'
        self.offline = offline

    def send_msg_impl(self, msg):
        sync = False if self.offline else True
        bos_send_msg(self, msg, sync)

    def recv_msg_impl(self):
        return bos_recv_msg(self)

    def send_tensor_impl(self, tensor):
        validation_log_info(self.get_tensor_file_path())
        torch.save(tensor, self.get_tensor_file_path())
        sync = False if self.offline else True
        bos_send_file(self, self.get_tensor_file_path(), sync)

    def recv_tensor_impl(self):
        tensor_path = sync_pull_file(self.address, self.get_tensor_file_name(), self.cache_dir)
        tensor = torch.load(tensor_path)
        exec_shell(f"rm -f {tensor_path}")
        return tensor


    def send_result_impl(self, result):
        if self.offline:
            return
        return self.send_msg_impl(result)

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

SOCKET_PORT = 1000
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

is_nvidia_gpu = os.environ.get('CUDA_VERSION', None) or os.environ.get('NVIDIA_VISIBLE_DEVICES', None)
validation_filter_port = os.environ.get('VALIDATION_FILTER_PORT', None)

class GpuValidation(TorchDispatchMode):
    '''
    master is the device will be validation
    client is golden device 
    '''
    def __init__(self, model_key, atol, rtol, address, port=SOCKET_PORT, cache_dir='/tmp', use_bos=True, fallback=False, offline=True):
        self.is_gpu = is_nvidia_gpu
        self.fallback = fallback
        self.offline = offline
        if not use_bos:
            master = socket_gpu_validation_master
            client = socket_gpu_validation_client
        else:
            master = bos_gpu_validation_master
            client = bos_gpu_validation_client
        if self.is_gpu:
            self.validation = client(model_key, atol, rtol, address, port, cache_dir, fallback, offline=offline)
            validation_log_debug(f"initialize client validation completed!")
        else:
            self.validation = master(model_key, atol, rtol, address, port, cache_dir, fallback, offline=offline)
            validation_log_debug(f"initialize master validation completed!")

    def __torch_dispatch__(self, op, types, dev_args=(), dev_kwargs=None):
        validation_log_info(f"will call op:{op}")

        result = op(*dev_args, **dev_kwargs)

        if validation_filter_port:
            if self.validation.port != int(validation_filter_port):
                return result
        
        if isinstance(result, torch.Tensor):
            self.validation.validate(result, f"{op}")
            if not self.offline or self.is_gpu:
                validation_log_info(f"validate op:{op} dtype:{result.dtype} shape:{result.shape} result:{self.validation.result}")
            if self.fallback and not self.is_gpu:
                self.validation.increase_index()
                return self.validation.golden_result
            self.validation.increase_index()
        else:
            validation_log_info(f'skip not tensor result:{type(result)} op:{op}')

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
    with GpuValidation(token, 1e-3, 1e-3, "", SOCKET_PORT, './'):
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
    with GpuValidation(token, 1e-3, 1e-3, "", SOCKET_PORT):
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
