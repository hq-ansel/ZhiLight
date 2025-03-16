#include "easyengine/core/exception.hh"
#include "easyengine/core/allocator.hh"
#include "easyengine/core/guard.hh"
#include "easyengine/logger/kernel_time_trace.hpp"
#include "easyengine/logger/std_log_op.hpp"
#include <vector>
#include <iostream>
#include <mutex>
#include <unistd.h>
#include <sys/syscall.h>

static int _get_tid() {
    return syscall(SYS_gettid);
}
// 将void*转换为unsigned long long
inline std::uintptr_t convert_uint(void* ptr) {
    return reinterpret_cast<std::uintptr_t>(ptr);
}
// 将unsigned long long转换为void*
inline void* convert_voidp(std::uintptr_t ptr) {
    return reinterpret_cast<void*>(ptr);
}

inline size_t round_up(size_t num, size_t multiple) {
    return (num + multiple - 1) / multiple * multiple;
}

namespace easyengine {

namespace core {

Memory MemoryAllocator::new_mem(int pos, void* ptr, size_t size) {
    // int pid = _get_tid();
    auto deleter = [=](void* ptr) {
        // EZ_ASSERT_EQ(pid, _get_tid(), "");
        this->free(ptr, size);
    };
    Memory ret = std::make_shared<Memory_>(ptr, virtual_dev_id, size, deleter);
    // 为什么是从begin加?不是从end加?
    mems.insert(mems.begin() + pos, std::weak_ptr<Memory_>(ret));
    used += size;
    peak = std::max(used, peak);
    return ret;
}
// 拷贝内存移动，原始地址的内存不进行手动处理
void MemoryAllocator::memory_move(void* dst, void* src, size_t nbytes) {
    std::uintptr_t ptr_dst = convert_uint(dst);
    std::uintptr_t ptr_src = convert_uint(src);
    // 目标地址在源地址之后，这可能在某些情况下导致数据损坏（例如，如果源和目标内存区域重叠）。
    if (ptr_dst > ptr_src)
        throw std::logic_error("memory move dst > src");
    //CUDA 库中定义的一个枚举值，表示内存拷贝的方向是从设备内存到设备内存。
    auto d2d = cudaMemcpyDeviceToDevice;
    // 检测拷贝后是否发生内存重叠
    bool overlap = ptr_dst + nbytes > ptr_src;
    auto gap_size = ptr_src - ptr_dst;
    // 如果内存重叠且 gap_size 较大（大于 mem_reserve / 2），将内存分成多个 gap_size 大小的块，逐个拷贝。
    if (overlap && gap_size >= mem_reserve / 2) {
        // std::cout << "Handle overlap1\n";
        for (size_t i = 0; i < (nbytes + gap_size - 1) / gap_size; ++i) {
            size_t piece = std::min(nbytes - i * gap_size, gap_size);
            auto src1 = (char*)src + i * gap_size;
            auto dst1 = (char*)dst + i * gap_size;
            EZ_CUDART_ASSERT(cudaMemcpyAsync(dst1, src1, piece, d2d, stream));
        }
    } else if (overlap) {
        // std::cout << "Handle overlap2\n";
        // 使用 move_buf 作为临时缓冲区，先将数据拷贝到 move_buf，再从 move_buf 拷贝到目标地址，避免数据覆盖。
        for (size_t i = 0; i < (nbytes + mem_reserve - 1) / mem_reserve; ++i) {
            size_t piece = std::min(nbytes - i * mem_reserve, mem_reserve);
            auto src1 = (char*)src + i * mem_reserve;
            auto dst1 = (char*)dst + i * mem_reserve;
            EZ_CUDART_ASSERT(cudaMemcpyAsync(move_buf, src1, piece, d2d, stream));
            EZ_CUDART_ASSERT(cudaMemcpyAsync(dst1, move_buf, piece, d2d, stream));
        }
    } else {
        EZ_CUDART_ASSERT(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, stream));
    }
}
// 返回整理后的内存指针位置
void* MemoryAllocator::defragmentation() {
    // easyengine::print_demangled_trace(15);
    DeviceGuard guard(dev_id);

    EZ_CUDART_ASSERT(cudaStreamSynchronize(stream));
    cudaEvent_t start, stop;
    logger::createStartEvent(true, &start, &stop, stream);
    auto last_ptr = convert_uint(base_ptr);
    for (size_t i = 0; i < mems.size(); ++i) {
        // 如果 std::weak_ptr 观察的对象仍然存在（即对象未被销毁）
        // lock 会返回一个 std::shared_ptr，增加对象的引用计数。
        Memory mem = mems[i].lock();
        if (mem == nullptr) {
            // tensor is freed in another thread
            EZ_ASSERT(mem != nullptr, "Memory was released unexpectedly");
            continue;
        }
        EZ_ASSERT(mem != nullptr, "Memory was released unexpectedly");
        auto ptr = convert_uint(mem->ptr);
        if (last_ptr < ptr) {
            memory_move(convert_voidp(last_ptr), mem->ptr, mem->num_bytes);
            mem->ptr = convert_voidp(last_ptr);
        }
        last_ptr += mem->num_bytes;
    }

    float elapsed_ms = logger::destroyDiffEvent(start, stop, stream);
    size_t freed = convert_uint(end_ptr) - last_ptr;
    if (std::getenv("EZ_DEBUG_LEVEL") != nullptr) {
        std::cout << "defragmentation: used=" << used / 1024 / 1024
            << "MB, freed=" << freed / 1024 / 1024
            << "MB, cost=" << elapsed_ms << "ms" << std::endl;
        std::cout << std::hex << "base_ptr=" << convert_uint(org_base_ptr)
            << ", end_ptr=" << convert_uint(end_ptr) << std::dec << std::endl;
    }

    return convert_voidp(last_ptr);
}
// 主体申请内存池
MemoryAllocator::MemoryAllocator(
    int dev_id, int virtual_dev_id, size_t memory_limit, cudaStream_t stream)
    : dev_id(dev_id), virtual_dev_id(virtual_dev_id),
     memory_limit(memory_limit), stream(stream) {

    char* print_env = std::getenv("EZ_PRINT_MEM_ALLOC_TIME");
    if (print_env != nullptr) {
        print_alloc_time_step = std::atoi(print_env);
    }

    // 通过dev_id获取设备属性
    cudaDeviceProp prop;
    EZ_CUDART_ASSERT(cudaGetDeviceProperties(&prop, dev_id));

    {
        DeviceGuard guard(dev_id);
        EZ_CUDART_ASSERT(cudaMalloc(&move_buf, mem_reserve));
        if (memory_limit > (20L << 30)) {
            memory_limit -= mem_reserve;
        }
        EZ_CUDART_ASSERT(cudaMalloc(&org_base_ptr, memory_limit));
        base_ptr = org_base_ptr;
        end_ptr = (char*)base_ptr + memory_limit;
        // std::cout << "BeginPtr:" << base_ptr << ", EndPtr:" << end_ptr << "\n";move_buf
    }
}

MemoryAllocator::MemoryAllocator(MemoryAllocator& p, size_t child_size, cudaStream_t stream)
    : dev_id(p.dev_id), virtual_dev_id(p.virtual_dev_id), memory_limit(child_size), stream(stream) {
    parent = &p;
    // EZ_ASSERT(p.mems.empty(), "parent allocator is busy.");
    // assert base_ptr<end_ptr - child_size
    EZ_ASSERT_LE((char*)p.base_ptr, (char*)p.end_ptr - child_size, "Not enough memory");
    move_buf = (char*)p.end_ptr - mem_reserve;
    memory_limit -= mem_reserve;
    end_ptr = move_buf;
    base_ptr = (char*)p.end_ptr - child_size;
    p.end_ptr = base_ptr;
    p.memory_limit -= child_size;
}

MemoryAllocator::~MemoryAllocator() {
    if (parent) {
        if (parent->end_ptr != base_ptr)
            std::cerr << "Free children not match parent end_ptr\n";
        parent->end_ptr = (char*) base_ptr + memory_limit + mem_reserve;
        parent->memory_limit += memory_limit + mem_reserve;
        return;
    }
    try {
        DeviceGuard guard(dev_id);
        EZ_CUDART_ASSERT(cudaFree(org_base_ptr));
        EZ_CUDART_ASSERT(cudaFree(move_buf));
    } catch (EngineException e) { std::cerr << e.what() << std::endl; }
}

void MemoryAllocator::freeze_model_memory() {
    if (mems.empty()) {
        return;
    }
    // EZ_ASSERT(frozen_mems.empty(), "freeze_model_memory should be called only once");
    // std::lock_guard 的析构函数会自动释放锁，因此不需要显式调用 mutex.unlock()。
    std::lock_guard<std::mutex> lock(mutex);
    // move base_ptr to free memory after model tensors
    base_ptr = defragmentation();
    // move mems(model tensors) to frozen_mems;
    // so they will not be searched again in the following alloc() calls.
    frozen_mems.insert(frozen_mems.end(), mems.begin(), mems.end());
    mems.clear();
}

Memory MemoryAllocator::alloc(size_t num_bytes, size_t round_up_bytes) {
    EZ_ASSERT(num_bytes > 0, "num_bytes must be greater than 0");
    EZ_ASSERT(round_up_bytes % 512 == 0, "round_up_bytes must be multiple of 512");

    static long count = 0;
    long start = print_alloc_time_step ? logger::get_time_us() : 0;

    num_bytes = round_up(num_bytes, round_up_bytes);

    std::lock_guard<std::mutex> lock(mutex);
    EZ_ASSERT_LE( num_bytes, memory_limit - used,
              logger::str_cat("Exceeded memory_limit:", memory_limit / 1024 / 1024, "MB"));

    auto last_ptr = convert_uint(base_ptr);
    for (size_t i = 0; i < mems.size(); ++i) {
        Memory mem = mems[i].lock();
        if (mem == nullptr) {
            // tensor freed in another thread
            // EZ_ASSERT(mem != nullptr, "Memory was released unexpectedly");
            continue;
        }
        // 如果申请小于当前内存块的剩余空间
        // 申请4K,但是新的块有10K 直接早停
        if (last_ptr + num_bytes <= convert_uint(mem->ptr)) {
            if (print_alloc_time_step && count++ % print_alloc_time_step == 0) {
                long time = logger::get_time_us() - start;
                std::cout << "Alloc1 take " << time << "us, mems=" << mems.size() << "\n";
            }
            return new_mem(i, convert_voidp(last_ptr), num_bytes);
        }
        last_ptr = convert_uint(mem->ptr) + mem->num_bytes;
    }

    // at the end
    if (last_ptr + num_bytes <= convert_uint(end_ptr)) {
        if (print_alloc_time_step && count++ % print_alloc_time_step == 0) {
            long time = logger::get_time_us() - start;
            std::cout << "Alloc2 take " << time << "us, mems=" << mems.size() << "\n";
        }
        return new_mem(mems.size(), convert_voidp(last_ptr), num_bytes);
    }

    if (!allow_gc_) {
        throw std::runtime_error("GC is disabled");
    }

    void* ptr = defragmentation();
    EZ_ASSERT(convert_uint(ptr) + num_bytes <= convert_uint(end_ptr),
              logger::str_cat("Exceeded memory_limit:", memory_limit / 1024 / 1024, "MB"));
    return new_mem(mems.size(), ptr, num_bytes);
}
// 删除指定块
void MemoryAllocator::free(void* ptr, size_t size) {
    // std::cout << "Free " << (size / 1000000) << "MB\n";
    std::lock_guard<std::mutex> lock(mutex);

    for (size_t i = 0; i < mems.size(); ++i) {
        Memory mem = mems[i].lock();
        if (mem == nullptr || mem->ptr == ptr) {
            mems.erase(mems.begin() + i);
            used -= size;
            return;
        }
    }
    for (int i = int(frozen_mems.size()) - 1; i >= 0; --i) {
        Memory mem = frozen_mems[i].lock();
        if (mem == nullptr || mem->ptr == ptr) {
            frozen_mems.erase(frozen_mems.begin() + i);
            used -= size;
            return;
        }
    }
    EZ_EXCEPTION("Memory was not allocated by this allocator");
}

size_t MemoryAllocator::used_memory() const {
    return used;
}

size_t MemoryAllocator::peak_memory() const {
    return peak;
}

DirectMemoryAllocator::DirectMemoryAllocator(
    int dev_id, int virtual_dev_id, size_t memory_limit, cudaStream_t stream)
    : MemoryAllocator(dev_id, virtual_dev_id, 1024, stream) {
    this->memory_limit = memory_limit;
    std::cout << "Use DirectMemoryAllocator" << std::endl;
}

Memory DirectMemoryAllocator::alloc(size_t size, size_t round_up_bytes) {
    if (round_up_bytes > 4096) {
        size_t old_size = size;
        size = round_up(size, round_up_bytes);
        // std::cout << "round up from " << old_size << " to " << size << ", round_up_bytes=" <<
        // round_up_bytes << std::endl;
    }
    void* ptr;
    {
        DeviceGuard guard(dev_id);
        EZ_CUDART_ASSERT(cudaMalloc(&ptr, size));
    }
    // std::cout << "dev:" << dev_id << " alloc " << ptr << ", size=" << size << std::endl;
    used += size;
    peak = std::max(used, peak);
    return std::make_shared<Memory_>(
        ptr, virtual_dev_id, size, [this, size](void* ptr) { this->free(ptr, size); });
}

void DirectMemoryAllocator::free(void* ptr, size_t size) {
    bool free_early = std::getenv("DIRECT_ALLOC_FREE_EARLY") != nullptr;
    if (free_early) {
        DeviceGuard guard(dev_id);
        EZ_CUDART_ASSERT(cudaDeviceSynchronize());
        EZ_CUDART_ASSERT(cudaFree(ptr));
    } else {
        frees.push_back(ptr);
    }
    used -= size;
}

void DirectMemoryAllocator::free_session() {
    DeviceGuard guard(dev_id);
    for (void* ptr : frees) {
        std::cerr << "dev:" << dev_id << " free " << ptr << std::endl;
        EZ_CUDART_ASSERT(cudaFree(ptr));
    }
    frees.clear();
}

} // namespace core

} // namespace easyengine
