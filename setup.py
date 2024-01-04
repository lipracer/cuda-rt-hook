from setuptools import setup
import os, sys

def sync_shell(cmd):
    err = os.system(cmd)
    assert not err, f'exec cmmand:{cmd} fail!'

sync_shell("rm -rf build")
script_dir = os.path.dirname(os.path.abspath(__file__))
sync_shell(f"cmake -S {script_dir} -B build && cmake --build build")
sync_shell(r"cd build/lib && ls | grep -v '\<cuda_mock\>' | xargs -I {} rm -rf {}")

setup()
