#pragma once
#include "easyengine/core/export.hh"
#include "easyengine/core/exception.hh"
#include <cuda_runtime.h>
#include <iostream>
namespace easyengine {

namespace core {

class Context;

class ENGINE_EXPORT WithDevice {
private:
    const Context* ctx;

public:
    WithDevice(const Context& ctx, int dev);
    ~WithDevice();
    WithDevice(const WithDevice&) = delete;
    WithDevice& operator=(const WithDevice&) = delete;
    WithDevice(WithDevice&&);
    WithDevice& operator=(WithDevice&&) = delete;
};

class ENGINE_EXPORT ScopeDevice {
private:
    Context* ctx;

public:
    ScopeDevice(const Context& ctx, int dev);
    ~ScopeDevice();
    ScopeDevice(const ScopeDevice&) = delete;
    ScopeDevice& operator=(const ScopeDevice&) = delete;
    ScopeDevice(ScopeDevice&&);
    ScopeDevice& operator=(ScopeDevice&&) = delete;
};

class ENGINE_EXPORT WithDebug {
private:
    const Context* ctx;
    const int previous_level;

public:
    WithDebug(const Context& ctx, int debug_level);
    ~WithDebug();
    WithDebug(const WithDebug&) = delete;
    WithDebug& operator=(const WithDebug&) = delete;
    WithDebug(WithDebug&&);
    WithDebug& operator=(WithDebug&&) = delete;
};
class DeviceGuard {
    private:
        int old_dev;
    
    public:
        DeviceGuard(int idx) {
            EZ_CUDART_ASSERT(cudaGetDevice(&old_dev));
            if (old_dev != idx) {
                EZ_CUDART_ASSERT(cudaSetDevice(idx));
            }
        }
        ~DeviceGuard() {
            try {
                if (old_dev != -1) {
                    EZ_CUDART_ASSERT(cudaSetDevice(old_dev));
                }
            } catch (const EngineException& e) { std::cerr << e.what() << std::endl; }
        }
        DeviceGuard(const DeviceGuard&) = delete;
        DeviceGuard(DeviceGuard&&) = delete;
    };
    
    template<typename T>
    class MemoryGuard {
    private:
        T* ptr;
    
    public:
        explicit MemoryGuard(T* p) : ptr(p) { }
        ~MemoryGuard() {
            try {
                if (ptr != nullptr) {
                    delete ptr;
                }
            } catch (const EngineException& e) { std::cerr << e.what() << std::endl; }
        }
        MemoryGuard(const MemoryGuard&) = delete;
        MemoryGuard(MemoryGuard&&) = delete;
    };
    
    template<typename T>
    class MemoryArrayGuard {
    private:
        T* ptr;
    
    public:
        explicit MemoryArrayGuard(T* p) : ptr(p) { }
        ~MemoryArrayGuard() {
            try {
                if (ptr != nullptr) {
                    delete[] ptr;
                }
            } catch (const EngineException& e) { std::cerr << e.what() << std::endl; }
        }
        MemoryArrayGuard(const MemoryArrayGuard&) = delete;
        MemoryArrayGuard(MemoryArrayGuard&&) = delete;
    };
}

}
