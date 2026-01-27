from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, library_paths

_rpaths = [f"-Wl,-rpath,{p}" for p in library_paths()]
_rpaths.append("-Wl,-rpath,$ORIGIN")

setup(
    name="ftic_rans",
    ext_modules=[
        CppExtension(
            name="ftic_rans",
            sources=["ftic_rans.cpp"],
            include_dirs=["."],
            extra_compile_args=["-O3", "-std=c++17"],
            extra_link_args=_rpaths,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
