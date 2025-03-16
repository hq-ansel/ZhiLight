#pragma once
#include <cuda_runtime.h>
#include <functional>
#include <memory>

namespace easyengine {

namespace core {
// 是一个通用的函数包装器，它可以存储、复制和调用任何可调用对象
//（如函数、lambda 表达式、函数对象等），
//这些可调用对象接受一个 cudaStream_t 类型的参数并返回 void。
typedef std::function<void(cudaStream_t)> StreamDeleter;

class Stream_ {
public:
    cudaStream_t ptr;
    StreamDeleter deleter;

    Stream_(cudaStream_t ptr, StreamDeleter deleter);
    Stream_(const Stream_&) = delete;
    Stream_(Stream_&&) = delete;
    Stream_& operator=(const Stream_&) = delete;
    Stream_& operator=(Stream_&&) = delete;
    ~Stream_();
};

typedef std::shared_ptr<Stream_> Stream;

}

}