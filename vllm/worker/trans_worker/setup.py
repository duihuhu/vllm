from setuptools import setup, Extension
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from torch.utils.cpp_extension import include_paths, library_paths

ext_modules = [
    Pybind11Extension(
        "trans",
        ["trans_engine.cc", "trans_config.cc", "trans_worker.cc", "binding.cc"],
        include_dirs=[pybind11.get_include(), 
            *include_paths(), 
            "/usr/local/cuda-12.2/targets/x86_64-linux/include/"],
        extra_compile_args=['-std=c++17'],
        libraries=['torch', 'c10'],
        library_dirs=[*library_paths() , "/usr/local/cuda/lib64"],
    ),
]

setup(
    name="trans",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
