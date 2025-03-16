#pragma once
#include "easyengine/core/export.hh"
#include "easyengine/core/memory.hh"
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <vector>

namespace easyengine {

namespace core {

class MemoryAllocator {
protected:
    int dev_id, virtual_dev_id;
    size_t memory_limit;
    cudaStream_t stream;
    // 使用弱引用，防止内存泄漏
    std::vector<std::weak_ptr<Memory_>> mems;
    // 模型的冻结内存向量
    std::vector<std::weak_ptr<Memory_>> frozen_mems;
    std::mutex mutex;

    size_t used { 0 };
    size_t peak { 0 };
    // cudaMalloc的返回值
    void *base_ptr, *org_base_ptr;
    // base_ptr + memory_limit
    void* end_ptr { nullptr };

    int print_alloc_time_step { 0 };
    // for memory_move() when memory areas are overlapped 暂定为64MB
    size_t mem_reserve { 64 * 1024 * 1024 };
    // move_buf = end_ptr - mem_reserve 类似栈的概念，用于临时存储数据
    void* move_buf;

    MemoryAllocator* parent { nullptr };

    bool allow_gc_ { true };

    Memory new_mem(int pos, void* ptr, size_t size);
    void memory_move(void* dst, void* src, size_t nbytes);

public:
    MemoryAllocator(const MemoryAllocator&) = delete;
    MemoryAllocator(MemoryAllocator&&) = delete;
    MemoryAllocator(int dev_id, int virtual_dev_id, size_t memory_limit, cudaStream_t stream);
    virtual ~MemoryAllocator();

    MemoryAllocator(MemoryAllocator& parent, size_t child_size, cudaStream_t stream);

    void* defragmentation();

    void freeze_model_memory();

    virtual Memory alloc(size_t num_bytes, size_t round_up_bytes = 1024);
    virtual void free(void* ptr, size_t size);

    virtual void free_session() { }

    size_t used_memory() const;
    size_t peak_memory() const;
    size_t get_memory_limit() const { return memory_limit; }
    size_t get_free_memory() const { return memory_limit - used_memory(); }
    size_t get_block_num() const { return mems.size(); }

    void set_allow_gc(bool allow) { allow_gc_ = allow; }
};

// for memory check
class DirectMemoryAllocator : public MemoryAllocator {
    std::vector<void*> frees;

public:
    DirectMemoryAllocator(int dev_id, int virtual_dev_id, size_t memory_limit, cudaStream_t stream);
    ~DirectMemoryAllocator() = default;

    Memory alloc(size_t num_bytes, size_t round_up_bytes) override;
    void free(void* ptr, size_t size) override;
    void free_session() override;
};

} // namespace core

} // namespace easyengine
