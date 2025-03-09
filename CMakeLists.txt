# 指定构建此项目所需的最低 CMake 版本
cmake_minimum_required(VERSION 3.20)

# 设置项目名称
project(zhilight)

# 启用 C 和 C++ 语言支持
enable_language(C)
enable_language(CXX)

# 尝试从环境变量中获取 CUDA 架构，如果未定义则设置为默认值 "80;89"
set(CMAKE_CUDA_ARCHITECTURES $ENV{CMAKE_CUDA_ARCHITECTURES})
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;89")
endif()
# 输出当前设置的 CUDA 架构
message("CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

# 启用 CUDA 语言支持
enable_language(CUDA)

# 设置 C++ 标准为 C++17，并要求必须使用该标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 设置 CUDA 标准为 C++17，并要求必须使用该标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# 设置 C++ 和 CUDA 的符号可见性预设
set(CMAKE_CXX_VISIBILITY_PRESET hidden)    # C++ 默认隐藏符号
set(CMAKE_CUDA_VISIBILITY_PRESET default) # CUDA 使用默认符号可见性

# 设置内联函数的隐藏属性（此处设置为不隐藏）
set(CMAKE_VISIBILITY_INLINES_HIDDEN 0)

# 设置构建时和安装时的运行路径为当前目录（使用 ${ORIGIN} 表示可执行文件所在目录）
set(CMAKE_BUILD_RPATH ${ORIGIN})
set(CMAKE_INSTALL_RPATH ${ORIGIN})

# 指定 CMake 模块的搜索路径
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

# 设置是否使用静态链接的 NCCL 库
set(USE_STATIC_NCCL True)

# 输出当前使用的 CUDA 工具包版本
message("CUDA Version: ${CMAKE_CUDA_COMPILER_TOOLKIT_VERSION}")

# 添加编译选项
add_compile_options(
    -Wno-deprecated-declarations        # 忽略弃用声明的警告
    -Wno-narrowing                      # 忽略窄化转换的警告
    # 针对 CUDA 编译器，抑制特定的诊断信息（错误号 177 和 549）
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=177,549">
)

# 查找 Python 及其开发和解释器组件，要求必须找到
find_package(Python COMPONENTS Interpreter Development REQUIRED)
# 设置 Python 可执行文件和版本的变量
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
set(PYTHON_VERSION ${Python_VERSION})

# 检查 CUDA 工具包版本是否低于 12.0.0，如果是则报错，否则输出信息
if(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION VERSION_LESS 12.0.0)
    error("Require CUDA 12")
else()
    message("Will link against CUDA 12 compiled libraries")
endif()

# 定义一个选项，默认启用单元测试
option(WITH_TESTING "Compile zhilight with unit testing" ON)

# 如果启用了单元测试
if(WITH_TESTING)
    # 使用 Python 获取 PyTorch 使用的 C++ ABI 版本，并存储在 PYTORCH_CXX_ABI 变量中
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import torch; \
            print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
        OUTPUT_VARIABLE PYTORCH_CXX_ABI
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    # 将 PyTorch 的 C++ ABI 版本作为编译定义添加到 C++ 和 CUDA 编译器标志中
    list(APPEND CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${PYTORCH_CXX_ABI}")
    list(APPEND CMAKE_CUDA_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${PYTORCH_CXX_ABI}")
    # 输出当前的 C++ 编译器标志
    message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
endif()

# 从 RELWITHDEBINFO 模式下的 C++ 和 CUDA 编译器标志中移除 "-DNDEBUG" 选项
string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string(REPLACE "-DNDEBUG" "" CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")

# 输出安装运行路径的信息
message(STATUS "CMAKE_INSTALL_RPATH: ${CMAKE_INSTALL_RPATH}")

# 包含自定义的 bmengine CMake 配置
include(bmengine)

# 添加子目录 "3rd/bmengine"，通常包含第三方库的构建和配置
add_subdirectory(3rd/bmengine)

# 使用 Python 获取 pybind11 的 CMake 目录，并存储在 pybind11_DIR 变量中
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; \
        print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# 查找 pybind11 包，要求必须找到
find_package(pybind11 REQUIRED)

# 递归地查找所有位于 src 目录下的 .cpp 和 .cu 文件，并存储在 BACKEND_SOURCES 变量中
file(GLOB_RECURSE BACKEND_SOURCES
    "src/*.cpp"
    "src/*.cu"
)

# 过滤掉不需要的源文件，排除 py_export 目录下的所有 .cpp 文件和 nn/tests 目录下的所有文件
list(FILTER BACKEND_SOURCES EXCLUDE REGEX "src/py_export/.*\\.cpp$")
list(FILTER BACKEND_SOURCES EXCLUDE REGEX "src/nn/tests/.*$")

# 创建一个名为 backend 的静态库，并将筛选后的源文件添加到其中
add_library(backend STATIC ${BACKEND_SOURCES})
# 设置 backend 库的编译属性，使生成的代码位置无关
set_property(TARGET backend PROPERTY POSITION_INDEPENDENT_CODE ON)

# 查找 Flash Attention 库的 .so 文件路径
file(GLOB FLASH_ATTN_LIB "${Python_SITELIB}/flash_attn*.so")
# 获取 Flash Attention 库文件的名称
get_filename_component(FLASH_ATTN_SO ${FLASH_ATTN_LIB} NAME)
# 输出 Flash Attention 库的名称
message("FLASH_ATTN_SO: " ${FLASH_ATTN_SO})

# 设置 backend 库的包含目录，供编译时使用
target_include_directories(backend
    PUBLIC "src"                                      # 项目 src 目录
    PUBLIC "3rd/tensorrt_llm/"                        # 第三方 TensorRT LLm 目录
    PUBLIC "3rd/flash_decoding"                       # 第三方 Flash Decoding 目录
    PUBLIC ${NCCL_INCLUDE_DIRS}                       # NCCL 包含目录
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}  # CUDA 工具包包含目录
    PUBLIC ${Python_SITELIB}/torch/include              # PyTorch 包含目录
)

# 设置 backend 库的链接目录，供链接时查找库文件
target_link_directories(backend
    PUBLIC "3rd/tensorrt_llm/"                        # 第三方 TensorRT LLm 库目录
    PUBLIC ${NCCL_LIBRARIES}                          # NCCL 库目录
    PUBLIC ${Python_SITELIB}/torch/lib                  # PyTorch 库目录
    PUBLIC ${Python_SITELIB}                          # Python 库目录
)

# 链接 backend 库所需的依赖库
target_link_libraries(backend
    PUBLIC bmengine                                   # 自定义的 bmengine 库
    "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions"          # 链接器选项，用于符号绑定
    ":${FLASH_ATTN_SO}"                               # 链接 Flash Attention 库
)

# 查找所有位于 src/py_export 目录下的 .cpp 文件，用于构建 pybind11 模块
file(GLOB_RECURSE PYBIND_SOURCES "src/py_export/*.cpp")

# 使用 pybind11 添加一个名为 C 的 Python 模块，并指定源文件
pybind11_add_module(C ${PYBIND_SOURCES})
# 为模块 C 添加编译定义，设置版本信息
target_compile_definitions(C
    PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# 链接模块 C 所需的库
target_link_libraries(C PRIVATE
    bmengine                                        # 自定义的 bmengine 库
    "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions"        # 链接器选项，用于符号绑定
    "pthread"                                       # POSIX 线程库
    "backend"                                       # 之前创建的 backend 静态库
)

# 如果启用了单元测试选项
if(WITH_TESTING)
    # 添加子目录 "tests/py_export_internal"，通常包含内部单元测试
    add_subdirectory(tests/py_export_internal)
    # 添加子目录 "src/nn/tests"，通常包含神经网络相关的单元测试
    add_subdirectory(src/nn/tests)
endif()