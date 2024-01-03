from setuptools import setup
import os

os.system("cmake -S . -B build && cmake --build build")
os.system("cd build/lib && ls | grep -v '\<cuda_mock\>' | xargs -I {} rm -rf {}")
setup()
