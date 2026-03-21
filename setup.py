from setuptools import setup, Extension
import pybind11
import sys

# Определяем флаги в зависимости от ОС
if sys.platform == 'win32':
    compile_args = ['-std=c++11']
else:
    compile_args = ['-std=c++11', '-O3', '-fPIC']

ext_modules = [
    Extension(
        'ih_coverage',
        ['src/pybind_wrapper.cpp', 'src/coverage_check.cpp'],
        include_dirs=[
            'include',
            pybind11.get_include()
        ],
        language='c++',
        extra_compile_args=compile_args,
    ),
]

setup(
    name='ih-coverage',
    version='0.1.0',
    description='Coverage check for IH library',
    ext_modules=ext_modules,
    zip_safe=False,
)
