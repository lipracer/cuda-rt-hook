from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
import subprocess
from setuptools.extension import Extension as _Extension
from glob import glob
import shutil

class CMakeExtension(_Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name))),
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DCMAKE_INSTALL_PREFIX=' + os.path.join(os.path.abspath(os.path.dirname(self.build_lib)), os.path.basename(self.build_lib))
        ]

        build_dir = os.path.abspath(os.path.join(self.build_temp, ext.name))
        os.makedirs(build_dir, exist_ok=True)

        ninja_args = [
            '-GNinja',  # Specify Ninja as the generator
        ]

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args + ninja_args, cwd=build_dir)
        subprocess.check_call(['cmake', '--build', '.'], cwd=build_dir)
        subprocess.check_call(['ninja', 'install'], cwd=build_dir)

setup(
    name="cuda-mock",
    version="0.0.9",
    author="lipracer",
    author_email="lipracer@gmail.com",
    description="a tools hook some api call at runtime",

    url="https://github.com/lipracer/torch-cuda-mock", 

    ext_modules=[CMakeExtension('cuda_mock_impl', sourcedir='.')],
    cmdclass=dict(build_ext=CMakeBuild),
    package_dir={'': 'src'},
    package_data={'' : ['*']},
    zip_safe=False,
)
