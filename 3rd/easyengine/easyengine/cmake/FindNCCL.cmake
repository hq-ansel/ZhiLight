# Find the nccl libraries
#
# The following variables are optionally searched for defaults
#  NCCL_ROOT: Base directory where all NCCL components are found
#  NCCL_INCLUDE_DIR: Directory where NCCL header is found
#  NCCL_LIB_DIR: Directory where NCCL library is found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIRS
#  NCCL_LIBRARIES
#
# The path hints include CUDA_TOOLKIT_ROOT_DIR seeing as some folks
# install NCCL in the same location as the CUDA toolkit.
# See https://github.com/caffe2/caffe2/issues/1601

# 设置 NCCL_INCLUDE_DIR 变量，用于指定包含 NCCL 头文件的目录
# 如果环境变量 NCCL_INCLUDE_DIR 存在，则使用其值，否则为空
set(NCCL_INCLUDE_DIR $ENV{NCCL_INCLUDE_DIR} CACHE PATH "Folder contains NVIDIA NCCL headers")

# 设置 NCCL_LIB_DIR 变量，用于指定包含 NCCL 库文件的目录
# 如果环境变量 NCCL_LIB_DIR 存在，则使用其值，否则为空
set(NCCL_LIB_DIR $ENV{NCCL_LIB_DIR} CACHE PATH "Folder contains NVIDIA NCCL libraries")

# 设置 NCCL_VERSION 变量，用于指定要使用的 NCCL 版本
# 如果环境变量 NCCL_VERSION 存在，则使用其值，否则为空
set(NCCL_VERSION $ENV{NCCL_VERSION} CACHE STRING "Version of NCCL to build with")

# 如果环境变量 NCCL_ROOT_DIR 存在，则发出警告，提示用户改用 NCCL_ROOT
if ($ENV{NCCL_ROOT_DIR})
  message(WARNING "NCCL_ROOT_DIR is deprecated. Please set NCCL_ROOT instead.")
endif()

# 将 NCCL_ROOT_DIR 和 CUDA_TOOLKIT_ROOT_DIR 添加到 NCCL_ROOT 列表中
# 用于在搜索路径中包含这些目录
list(APPEND NCCL_ROOT $ENV{NCCL_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR})

# 将 NCCL_ROOT 添加到 CMAKE_PREFIX_PATH 中
# 这是为了兼容 CMake <3.12 版本，确保 NCCL_ROOT 被包含在搜索路径中
list(APPEND CMAKE_PREFIX_PATH ${NCCL_ROOT})

# 注释掉的示例路径，用于手动指定 NCCL 头文件和库文件的路径
#set(NCCL_INCLUDE_DIR /usr/local/cuda/include)
#set(NCCL_LIB_DIR /usr/local/cuda/lib64)

# 查找 NCCL 头文件路径，优先使用 NCCL_INCLUDE_DIR 作为提示路径
find_path(NCCL_INCLUDE_DIRS
  NAMES nccl.h
  HINTS ${NCCL_INCLUDE_DIR})

# 根据 USE_STATIC_NCCL 变量决定链接静态库还是动态库
if (USE_STATIC_NCCL)
  MESSAGE(STATUS "USE_STATIC_NCCL is set. Linking with static NCCL library.")
  SET(NCCL_LIBNAME "nccl_static")  # 静态库名
  if (NCCL_VERSION)  # 如果指定了 NCCL 版本，则优先查找版本化的静态库
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
else()
  SET(NCCL_LIBNAME "nccl")  # 动态库名
  if (NCCL_VERSION)  # 如果指定了 NCCL 版本，则优先查找版本化的动态库
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
endif()

# 查找 NCCL 库文件路径，优先使用 NCCL_LIB_DIR 作为提示路径
find_library(NCCL_LIBRARIES
  NAMES ${NCCL_LIBNAME}
  HINTS ${NCCL_LIB_DIR})

# 使用标准参数处理函数检查 NCCL 是否找到
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIRS NCCL_LIBRARIES)

# 如果 NCCL 找到，则进行版本检查和验证
if(NCCL_FOUND)
  # 设置 NCCL 头文件路径
  set (NCCL_HEADER_FILE "${NCCL_INCLUDE_DIRS}/nccl.h")
  message (STATUS "Determining NCCL version from ${NCCL_HEADER_FILE}...")

  # 保存当前的 CMAKE_REQUIRED_INCLUDES，以便后续恢复
  set (OLD_CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES})
  list (APPEND CMAKE_REQUIRED_INCLUDES ${NCCL_INCLUDE_DIRS})

  # 检查 NCCL 头文件中是否定义了 NCCL_VERSION_CODE
  include(CheckCXXSymbolExists)
  check_cxx_symbol_exists(NCCL_VERSION_CODE nccl.h NCCL_VERSION_DEFINED)

  # 如果 NCCL_VERSION_CODE 存在，则进一步验证版本
  if (NCCL_VERSION_DEFINED)
    # 创建一个临时 CUDA 文件，用于提取 NCCL 版本
    set(file "${PROJECT_BINARY_DIR}/detect_nccl_version.cu")
    file(WRITE ${file} "
      #include <iostream>
      #include <nccl.h>
      int main()
      {
        std::cout << NCCL_MAJOR << '.' << NCCL_MINOR << '.' << NCCL_PATCH << std::endl;

        int x;
        ncclGetVersion(&x);
        return x == NCCL_VERSION_CODE;
      }
")
    # 编译并运行临时文件，获取 NCCL 版本
    try_run(NCCL_VERSION_MATCHED compile_result ${PROJECT_BINARY_DIR} ${file}
          RUN_OUTPUT_VARIABLE NCCL_VERSION_FROM_HEADER
          CMAKE_FLAGS  "-DINCLUDE_DIRECTORIES=${NCCL_INCLUDE_DIRS}"
          LINK_LIBRARIES ${NCCL_LIBRARIES})

    # 如果头文件和库文件的版本不匹配，则报错
    if (NOT NCCL_VERSION_MATCHED)
      message(FATAL_ERROR "Found NCCL header version and library version do not match! \
(include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES}) Please set NCCL_INCLUDE_DIR and NCCL_LIB_DIR manually.")
    endif()
    # 输出 NCCL 版本
    message(STATUS "NCCL version: ${NCCL_VERSION_FROM_HEADER}")
  else()
    # 如果 NCCL_VERSION_CODE 不存在，则说明版本较旧
    message(STATUS "NCCL version < 2.3.5-5")
  endif ()

  # 恢复 CMAKE_REQUIRED_INCLUDES
  set (CMAKE_REQUIRED_INCLUDES ${OLD_CMAKE_REQUIRED_INCLUDES})

  # 输出 NCCL 的最终配置信息
  message(STATUS "Found NCCL (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES})")

  # 将 NCCL_ROOT_DIR、NCCL_INCLUDE_DIRS 和 NCCL_LIBRARIES 标记为高级变量
  mark_as_advanced(NCCL_ROOT_DIR NCCL_INCLUDE_DIRS NCCL_LIBRARIES)
endif()