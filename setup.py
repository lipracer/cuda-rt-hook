from setuptools import setup
import os

def sync_shell(cmd):
    err = os.system(cmd)
    assert not err, f'exec cmmand:{cmd} fail!'

sync_shell("rm -rf build")
sync_shell("cmake -S . -B build -GNinja && cmake --build build")
sync_shell("cd build/lib && ls | grep -v '\<cuda_mock\>' | xargs -I {} rm -rf {}")
setup()
