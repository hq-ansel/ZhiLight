#pragma once
#include "easyengine/core/export.hh"
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>

// 定义宏用于优化条件分支的预测
#if defined(__GNUC__)
// 使用GCC内置函数优化条件分支，表示条件很可能为真
static inline bool _likely(bool x) {
    return __builtin_expect((x), true);
}
// 使用GCC内置函数优化条件分支，表示条件很可能为假
static inline bool _unlikely(bool x) {
    return __builtin_expect((x), false);
}
#else
// 对于非GCC编译器，直接返回条件值
static inline bool _likely(bool x) {
    return x;
}
static inline bool _unlikely(bool x) {
    return x;
}
#endif

// 自定义异常类，继承自std::runtime_error
class EngineException : public std::runtime_error {
    const char* file;  // 异常发生的文件名
    int line;          // 异常发生的行号
    const char* func;  // 异常发生的函数名
    std::string info;  // 额外的异常信息
    std::string error_msg;  // 完整的错误信息

public:
    // 构造函数，初始化异常信息
    EngineException(
        const std::string& msg,
        const char* file_,
        int line_,
        const char* func_,
        const std::string& info_ = "")
        : std::runtime_error(msg), file(file_), line(line_), func(func_), info(info_) {
#ifdef NDEBUG
        // 在发布模式下，仅显示错误信息和额外信息
        error_msg = msg + "\n" + info_ + "\n\n";
#else
        // 在调试模式下，显示文件名、行号、函数名、错误信息和额外信息
        error_msg = std::string("File: ") + file_ + ":" + std::to_string(line_) + " " + func_ + "\n"
                    + msg + "\n" + info_ + "\n\n";
#endif
        // 在异常未被捕获时，将错误信息输出到标准错误流
        std::cerr << error_msg << std::endl;
    }

    // 重写what()函数，返回完整的错误信息
    const char* what() const noexcept { return error_msg.c_str(); }
};

// 断言宏，如果条件为假，抛出异常并输出堆栈跟踪
#define EZ_ASSERT(cond, msg)                                                                       \
    if (_unlikely(!(cond))) {                                                                    \
        easyengine::print_demangled_trace(15);                                                       \
        throw EngineException(                                                                   \
            "Assertion failed: " #cond, __FILE__, __LINE__, __PRETTY_FUNCTION__, msg);             \
    }

// 断言宏，如果x不等于y，抛出异常并输出堆栈跟踪
#define EZ_ASSERT_EQ(x, y, msg)                                                                    \
    if (_unlikely((x) != (y))) {                                                                 \
        easyengine::print_demangled_trace(15);                                                       \
        throw EngineException(                                                                   \
            "Assertion failed: " #x " != " #y " i.e. " + std::to_string(x)                         \
                + " != " + std::to_string(y),                                                      \
            __FILE__,                                                                              \
            __LINE__,                                                                              \
            __PRETTY_FUNCTION__,                                                                   \
            msg);                                                                                  \
    }

// 断言宏，如果x不小于y，抛出异常并输出堆栈跟踪
#define EZ_ASSERT_LT(x, y, msg)                                                                    \
    if (_unlikely((x) >= (y))) {                                                                 \
        easyengine::print_demangled_trace(15);                                                       \
        throw EngineException(                                                                   \
            "Assertion failed: " #x " < " #y " i.e. " + std::to_string(x)                         \
                + " < " + std::to_string(y),                                                      \
            __FILE__,                                                                              \
            __LINE__,                                                                              \
            __PRETTY_FUNCTION__,                                                                   \
            msg);                                                                                  \
    }

// 断言宏，如果x大于y，抛出异常并输出堆栈跟踪
#define EZ_ASSERT_LE(x, y, msg)                                                                    \
    if (_unlikely((x) > (y))) {                                                                 \
        easyengine::print_demangled_trace(15);                                                       \
        throw EngineException(                                                                   \
            "Assertion failed: " #x " <= " #y " i.e. " + std::to_string(x)                         \
                + " <= " + std::to_string(y),                                                      \
            __FILE__,                                                                              \
            __LINE__,                                                                              \
            __PRETTY_FUNCTION__,                                                                   \
            msg);                                                                                  \
    }

// CUDA运行时错误检查宏，如果status不为cudaSuccess，抛出异常并输出堆栈跟踪
#define EZ_CUDART_ASSERT(status)                                                                   \
    do {                                                                                           \
        cudaError_t _v = (status);                                                                  \
        if (_unlikely(_v != cudaSuccess)) {                                                       \
            easyengine::print_demangled_trace(15);                                                   \
            throw EngineException(                                                               \
                "CUDA Runtime Error: " #status,                                                    \
                __FILE__,                                                                          \
                __LINE__,                                                                          \
                __PRETTY_FUNCTION__,                                                               \
                cudaGetErrorString(_v));                                                            \
        }                                                                                          \
    } while (0)

// CUBLAS错误检查宏，如果status不为CUBLAS_STATUS_SUCCESS，抛出异常并输出堆栈跟踪
#define EZ_CUBLAS_ASSERT(status)                                                                   \
    do {                                                                                           \
        cublasStatus_t v = (status);                                                               \
        if (_unlikely(v != CUBLAS_STATUS_SUCCESS)) {                                             \
            easyengine::print_demangled_trace(15);                                                   \
            throw EngineException(                                                               \
                "CUBLAS Error: " #status,                                                          \
                __FILE__,                                                                          \
                __LINE__,                                                                          \
                __PRETTY_FUNCTION__,                                                               \
                cublasGetErrorString(v));                                                          \
        }                                                                                          \
    } while (0)

// NCCL错误检查宏，如果status不为ncclSuccess，抛出异常并输出堆栈跟踪
#define EZ_NCCL_ASSERT(status)                                                                     \
    do {                                                                                           \
        ncclResult_t v = (status);                                                                 \
        if (_unlikely(v != ncclSuccess))                                                         \
            throw EngineException(                                                               \
                "NCCL Error: " #status,                                                            \
                __FILE__,                                                                          \
                __LINE__,                                                                          \
                __PRETTY_FUNCTION__,                                                               \
                ncclGetErrorString(v));                                                            \
    } while (0)

// 抛出异常宏，直接抛出EngineException异常
#define EZ_EXCEPTION(msg)                                                                          \
    throw EngineException("Exception:\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, msg)

// CURAND错误检查宏，如果err不为CURAND_STATUS_SUCCESS，抛出异常
#define EZ_CURAND_CHECK(err)                                                                          \
    do {                                                                                           \
        curandStatus_t err_ = (err);                                                               \
        if (err_ != CURAND_STATUS_SUCCESS) {                                                       \
            throw EngineException(                                                               \
                "Exception:\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, "curand error");          \
        }                                                                                          \
    } while (0)

namespace easyengine {
BMENGINE_EXPORT void backtrace(int depth);  // 输出堆栈跟踪
BMENGINE_EXPORT void print_demangled_trace(int depth);  // 输出解构后的堆栈跟踪
BMENGINE_EXPORT const char* cublasGetErrorString(cublasStatus_t status);  // 获取CUBLAS错误信息字符串
} // namespace easyengine