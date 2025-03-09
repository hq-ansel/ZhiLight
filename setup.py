import os
import re
import subprocess
import sys

from typing import List
from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext

ROOT_DIR = os.path.dirname(__file__)
# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    """自定义的CMake扩展类，继承自Extension类"""
    
    def __init__(self, name, target="C", sourcedir=""):
        """初始化CMakeExtension实例
        
        参数:
        name : str
            扩展的名称
        target : str
            目标语言，默认为"C"
        sourcedir : str
            源代码目录，默认为空
        """
        Extension.__init__(self, name, sources=[])
        self.target = target
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    '''
    from setuptools.command.build_ext import build_ext
    build_ext 是setuptools的build_ext命令，继承自distutils.command.build_ext。
    distutils.command.build_ext 是Python的默认构建系统，负责编译Python扩展模块。
    这个类的作用是继承build_ext，并重写其build_extension方法，以调用CMake编译器编译C++扩展模块。
    distutils.command.build_ext 的run方法在配置完参数会调用build_extension方法编译扩展模块。   | 这里的self是setuptools.command.build_ext类
    CmakeBuild.run()->setuptools.command.build_ext.run()->distutils.command.build_ext.run(self)->
    distutils.command.build_ext.build_extension()
    '''
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "RelWithDebInfo" if debug else "Release"
        _testing = int(os.environ.get("TESTING", 1))
        testing_cfg = "ON" if _testing else "OFF"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            # 指定C++标准为C++17，确保编译器使用C++17标准来编译源文件。
            "-DCMAKE_CXX_STANDARD=17",
            
            # 设置输出库文件的目录为扩展模块的目录（extdir）。
            # 这个目录是最终生成的共享库或静态库的目标位置。
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            
            # 指定Python可执行文件的路径。这个值通常指向当前使用的Python解释器，
            # 以确保生成的扩展与该Python版本兼容。
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            
            # 指定Python的版本号，格式为主版本号.次版本号（例如3.12）。
            # 这个信息可能被用于依赖管理和兼容性检查。
            f"-DPYTHON_VERSION={sys.version_info.major}.{sys.version_info.minor}",
            
            # 设置CMake的构建类型（如Release, Debug等）。虽然在MSVC（Microsoft Visual C++）中不使用这个变量，
            # 但在其他编译器环境下是有用的，因此这里设置也不会造成负面影响。
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
            
            # 启用或禁用测试配置。根据项目的具体需求，可能需要开启或关闭单元测试等相关功能。
            f"-DWITH_TESTING={testing_cfg}",
        ]

        # zhilight can be compiled with various versions of g++ and CUDA,
        # only use standard toolchain in packaging container.
        if os.path.exists("/opt/rh/devtoolset-7/root/bin/gcc"):
            cmake_args.extend(
                [
                    "-DCMAKE_C_COMPILER=/opt/rh/devtoolset-7/root/bin/gcc",
                    "-DCMAKE_CXX_COMPILER=/opt/rh/devtoolset-7/root/bin/g++",
                    "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.1/bin/nvcc",
                ]
            )
        build_args = [f"--target {ext.target}"]
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]

        # Using Ninja-build since it a) is available as a wheel and b)
        # multithreads automatically. MSVC would require all variables be
        # exported for Ninja to pick it up, which is a little tricky to do.
        # Users can override the generator with CMAKE_GENERATOR in CMake
        # 3.15+.
        if not cmake_generator or cmake_generator == "Ninja":
            try:
                import ninja  # noqa: F401

                ninja_executable_path = os.path.join(ninja.BIN_DIR, "ninja")
                cmake_args += [
                    "-GNinja",
                    f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                ]
            except ImportError:
                pass

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        cmake_args += [
            "-DPython_ROOT_DIR=" + os.path.dirname(os.path.dirname(sys.executable))
        ]
        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)


ext_modules = [
    CMakeExtension("zhilight.C", "C"),
]

testing = int(os.environ.get("TESTING", 1))

if testing:
    ext_modules.append(CMakeExtension("zhilight.internals_", "internals_"))

# 移步version.py里去修改__version__
from version import __version__

def get_path(*filepath) -> str:
    """
    在file的中定义了全局变量ROOT_DIR，将filepath与ROOT_DIR拼接，返回绝对路径
    """
    return os.path.join(ROOT_DIR, *filepath)

def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif line.startswith("--"):
                continue
            else:
                resolved_requirements.append(line)
        return resolved_requirements
    return _read_requirements("requirements.txt")

setup(
    name="zhilight",
    version=__version__,
    author="Zhihu and ModelBest Teams",
    description="Optimized inference engine for llama and similar models",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    packages= find_packages(exclude=("tests", )),
    python_requires=">=3.9",
    include_package_data=True,
    install_requires=get_requirements(),
)
