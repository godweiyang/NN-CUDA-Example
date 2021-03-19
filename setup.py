from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="add2",
    ext_modules=[
        CppExtension(
            "add2",
            ["kernel/add2.cpp", "kernel/add2.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)