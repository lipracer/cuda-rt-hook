from setuptools import setup
import os

build_cpp_dir = 'build_cpp'
build_py_dir = 'build'

os.system(f"cmake -S . -B {build_cpp_dir}")
os.system(f"cmake --build {build_cpp_dir}")

os.system(f"mkdir -p {build_py_dir}/lib/cuda_mock")
os.system(f"cp {build_cpp_dir}/lib/cuda_mock/*.so {build_py_dir}/lib/cuda_mock")

setup()