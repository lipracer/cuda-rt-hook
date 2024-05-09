from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
import subprocess
from setuptools.extension import Extension as _Extension
from glob import glob
import shutil

script_dir = os.path.dirname(os.path.abspath(__file__))

publish_build = False
if os.environ.get('PUBLISH_BUILD'):
    publish_build = True

class CMakeExtension(_Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):

        ninja_args = []
        install_cmd = 'make'
        if not publish_build:
            ninja_args = ['-GNinja']
            install_cmd = 'ninja'

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name))),
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DCMAKE_INSTALL_PREFIX=' + os.path.join(os.path.abspath(os.path.dirname(self.build_lib)), os.path.basename(self.build_lib)),
            '-DENABLE_BUILD_WITH_GTEST=OFF',
            # f'-B {os.path.join(script_dir, "build")}',
        ]
        if publish_build:
            cmake_args.append('-DPYTHON_INCLUDE_DIR=' + os.environ.get('PYTHON_INCLUDE_DIR'))
            cmake_args.append('-DPYTHON_LIBRARY=' + os.environ.get('PYTHON_LIBRARY'))
            cmake_args.append('-DPUBLISH_BUILD=ON')

        build_dir = os.path.abspath(os.path.join(self.build_temp, ext.name))
        os.makedirs(build_dir, exist_ok=True)

        subprocess.check_call(['cmake', f'{script_dir}'] + cmake_args + ninja_args, cwd=build_dir)
        subprocess.check_call(['cmake', '--build', '.'], cwd=build_dir)
        subprocess.check_call([f'{install_cmd}', 'install'], cwd=build_dir)

setup(
    name="cuda-mock",
    version="0.1.9",
    author="lipracer",
    author_email="lipracer@gmail.com",
    description="a tools hook some api call at runtime",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    url="https://github.com/lipracer/torch-cuda-mock",

    ext_modules=[CMakeExtension('cuda_mock_impl', sourcedir='.')],
    cmdclass=dict(build_ext=CMakeBuild),
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    package_data={'' : ['*']},
    zip_safe=False,
)
