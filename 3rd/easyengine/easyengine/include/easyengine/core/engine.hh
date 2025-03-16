#pragma once
#include "easyengine/core/engine.hh"
#include "easyengine/core/thread_pool.hh"
#include "easyengine/core/allocator.hh"
#include "easyengine/core/stream.hh"
#include <mutex>
#include <stack>
#include <nccl.h>
#include <cublasLt.h>
#include <cuda_runtime.h>


namespace easyengine {

namespace core {
class Context;
class MemoryAllocator;
class DataType;

struct DeviceConfiguration {
    int device_id;
    size_t memory_limit;

    DeviceConfiguration(int device_id, size_t memory_limit)
        : device_id(device_id), memory_limit(memory_limit) { }
};

struct GPUInfo {
    int real_device_idx;
    int compute_capability;
    size_t total_memory;
    size_t free_memory;
    size_t alloc_memory;
};

class DeviceHandles {
public:
    int dev_id;
    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    ncclComm_t comm;
    int rank;
    int compute_capability;
    int mp_count;
    int l2_cache_size;
    int max_shared_memory;

    DeviceHandles(int dev_id, ncclUniqueId uniqueID, int rank, int world_size);
    ~DeviceHandles();
    DeviceHandles(const DeviceHandles&) = delete;
    DeviceHandles& operator=(const DeviceHandles&) = delete;
    DeviceHandles(DeviceHandles&&) = default;
    DeviceHandles& operator=(DeviceHandles&&) = delete;
};

class Tensor;
class Context;
class EngineImpl {
    friend class Engine;
    std::vector<DeviceHandles*> handles;
    std::vector<MemoryAllocator*> allocators;
    std::vector<StreamAllocator*> streams;
    std::vector<std::mutex*> device_lock;
    std::vector<TaskThreadPool*> device_threads;
    // for nccl
    std::vector<ncclUniqueId> uniqueIDs;
    int world_size_;

    int debug;
    bool is_mem_frozen { false };

public:
    EngineImpl(const std::vector<DeviceConfiguration>& cfg, int tp);
    ~EngineImpl();
    EngineImpl(const EngineImpl&) = delete;
    EngineImpl& operator=(const EngineImpl&) = delete;
    EngineImpl(EngineImpl&&) = delete;
    EngineImpl& operator=(EngineImpl&&) = delete;

    Context create_context(const std::vector<int>& devices) const;

    /* Thread-safe API */
    DeviceHandles* get_device_handle(int dev_id);
    void alloc_device(int dev_id);
    void release_device(int dev_id);
    cudaStream_t create_stream(int dev_id);
    void destroy_stream(int dev_id, cudaStream_t stream);

    MemoryAllocator* get_allocator(int dev_id) {
        return allocators[dev_id];
    }
    Memory alloc_memory(int dev_id, size_t size, size_t round_up_bytes = 512);
    Tensor alloc_tensor(int dev_id, const std::vector<size_t>& shape, DataType dtype);
    void get_parameter(const std::string& name, Tensor* tensor);
    void init_parameter(const std::string& name, Tensor* tensor);

    GPUInfo get_gpu_info(int dev_id);
    int num_gpus() const;
    int world_size() const { return world_size_; }

    void print_memory_summary();
    void freeze_model_memory();

    void device_foreach(std::function<void(int)>& fn);
    std::mutex log_mutex;
};
// Engine can be accessed from multiple threads.
class ENGINE_EXPORT Engine {
    // friend class DistributedTensorImpl;
    std::unique_ptr<EngineImpl> pimpl;

public:
    Engine(const std::vector<DeviceConfiguration>& cfg, int tp = 0);
    ~Engine();

    Context create_context(const std::vector<int>& devices) const;
    Context create_context() const; // use all devices
    Context create_context_rank(int rank) const;
    int num_gpus() const;
    int world_size() const;
    GPUInfo get_gpu_info(int device_idx) const;

    // Disable copy
    Engine(const Engine&) = delete;
    Engine(Engine&&) = delete;

    void device_foreach(std::function<void(int)> fn);
    void print_memory_summary();
    void freeze_model_memory();
    MemoryAllocator* get_allocator(int dev_id);

};

} // namespace core

} // namespace easyengine
