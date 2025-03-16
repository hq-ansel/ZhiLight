#include "easyengine/core/exception.hh"
#include <cstdio>
#include <execinfo.h>
#include <cxxabi.h>

namespace easyengine {

// 根据 CUBLAS 状态码返回对应的错误信息字符串
const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "Unknown CUBLAS error";
    }
}

// 输出当前调用堆栈的地址信息到标准错误流
void backtrace(int depth) {
    void** array = new void*[depth];  // 分配数组用于存储堆栈地址
    size_t size;
    // 获取当前堆栈的所有地址信息
    size = ::backtrace(array, depth);
    // 将堆栈地址信息输出到标准错误流
    backtrace_symbols_fd(array, size, fileno(stderr));
    delete[] array;  // 释放数组内存
}

// https://www.boost.org/doc/libs/1_65_1/boost/stacktrace/stacktrace.hpp
// std::cout << boost::stacktrace::stacktrace() << std::endl;
// 输出解构后的调用堆栈信息到标准错误流
void print_demangled_trace(int max_frames) {
    FILE* out = stderr;  // 输出到标准错误流
    fprintf(out, "[stack trace]:\n");

    // 存储堆栈地址的数组
    void* addrlist[max_frames + 1];

    // 获取当前堆栈地址信息
    int addrlen = ::backtrace(addrlist, max_frames);

    if (addrlen == 0) {
        fprintf(out, "  <empty, possibly corrupt>\n");  // 如果堆栈为空，输出提示信息
        return;
    }

    // 将地址信息解析为包含 "文件名(函数名+地址)" 的字符串数组
    char** symbollist = backtrace_symbols(addrlist, addrlen);

    // 分配用于存储解构后的函数名的字符串
    size_t funcnamesize = 512;
    char* funcname = (char*) malloc(funcnamesize);

    // 遍历堆栈地址信息，跳过第一个地址（当前函数）
    for (int i = 1; i < addrlen; i++) {
        char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

        // 解析堆栈字符串，找到函数名和偏移量的起始和结束位置
        for (char* p = symbollist[i]; *p; ++p) {
            if (*p == '(')
                begin_name = p;
            else if (*p == '+')
                begin_offset = p;
            else if (*p == ')' && begin_offset) {
                end_offset = p;
                break;
            }
        }

        if (begin_name && begin_offset && end_offset && begin_name < begin_offset) {
            *begin_name++ = '\0';  // 分隔出文件名
            *begin_offset++ = '\0';  // 分隔出函数名
            *end_offset = '\0';  // 分隔出偏移量

            // mangled name is now in [begin_name, begin_offset) and caller
            // offset in [begin_offset, end_offset). now apply
            // 使用 __cxa_demangle 解构函数名
            int status;
            char* ret = abi::__cxa_demangle(begin_name, funcname, &funcnamesize, &status);
            if (status == 0) {
                funcname = ret;  // 使用解构后的函数名
                fprintf(out, "  %s : %s +%s\n", symbollist[i], funcname, begin_offset);
            } else {
                // 如果解构失败，以 C 函数形式输出函数名
                fprintf(out, "  %s : %s() +%s\n", symbollist[i], begin_name, begin_offset);
            }
        } else {
            // 如果无法解析堆栈字符串，直接输出原始字符串
            fprintf(out, "  %s\n", symbollist[i]);
        }
    }

    free(funcname);  // 释放函数名字符串内存
    free(symbollist);  // 释放堆栈字符串数组内存
}

} // namespace easyengine