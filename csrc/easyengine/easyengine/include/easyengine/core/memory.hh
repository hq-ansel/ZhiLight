#pragma once
#include <functional>
#include <memory>
#include <vector>

namespace easyengine {
namespace core {
// 定义了一个函数指针，用于删除内存
typedef std::function<void(void*)> MemoryDeleter;

// 定义了一个类，用于管理内存
class Memory_ {
public:
    void* ptr;
    int dev;
    size_t num_bytes;
    std::vector<MemoryDeleter> deleters;

    Memory_(void* ptr, int dev, size_t num_bytes, MemoryDeleter deleter);
    Memory_(const Memory_&) = delete;
    Memory_(Memory_&&) = default;
    Memory_& operator=(const Memory_&) = delete;
    Memory_& operator=(Memory_&&) = default;
    ~Memory_();

    void add_deleter(MemoryDeleter deleter);
};

// 定义了一个智能指针，用于管理内存
typedef std::shared_ptr<Memory_> Memory;

}
}