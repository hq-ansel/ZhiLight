
与tensor类相关的文件

core/tensor_impl.cpp // pimpl的具体实现
core/tensor_print.cpp // tensor的打印函数
core/tensor.cpp // tensor类的构造函数、析构函数、拷贝构造函数、赋值运算符

include/easyengine/allocator.hh 
// MemoryAllocator类  DirectMemoryAllocator类
include/easyengine/context.hh 
// Context EventScope GCStopper
include/easyengine/context_impl.hh 
// ContextImpl类 EventRecord结构体
include/easyengine/dtype.hh 
// DataType DTypeDeducer
include/easyengine/engine.hh 
//DeviceConfiguration类  DeviceHandles类 EngineImpl类 Engine类
include/easyengine/guard.hh 
// WithDevice ScopeDevice WithDebug  MemoryGuard MemoryArrayGuard
include/easyengine/memory.hh
// Memory_类 共享指针管理 MemoryDeleter
include/easyengine/stream.hh
// Stream_类 共享指针管理 StreamDeleter 
include/easyengine/tensor.hh 
// tensor类 
include/easyengine/tensor_impl.hh 
// TensorImpl类
include/easyengine/thread_pool.hh 
// ThreadPool类

大概层次 从高到低

Engine(EngineImpl)
WithDevice WithDebug 
Context(ContextImpl)  EventScope GCStopper
Tensor(TensorImpl)   EventRecord  
  
MemoryAllocator DirectMemoryAllocator
DataType Memory GPUInfo Stream
MemoryGuard MemoryArrayGuard
DeviceGuard DeviceHandles DeviceConfiguration
EngineException
Memory_ 
ncclComm_t cublasHandle_t
cudaEvent_t 
cudaStream_t
cudaDeviceProp



Memory MemoryAllocator::new_mem(int pos, void* ptr, size_t size)
申请指定大小内存块,从池化内存中分配一段地址得到内存块,
插入内存块列表

void MemoryAllocator::memory_move(void* dst, void* src, size_t nbytes)
拷贝内存移动，非显式释放内存,直接改动内存块的指向地址

void* MemoryAllocator::defragmentation()
内存碎片整理，尝试将每个块向内存池头部移动

MemoryAllocator::MemoryAllocator(
    int dev_id, int virtual_dev_id, size_t memory_limit, cudaStream_t stream)
    : dev_id(dev_id), virtual_dev_id(virtual_dev_id),
     memory_limit(memory_limit), stream(stream) 
分配器先分配一大段池化内存,设置成员变量,与流句柄

MemoryAllocator::MemoryAllocator(MemoryAllocator& p, size_t child_size, cudaStream_t stream)
    : dev_id(p.dev_id), virtual_dev_id(p.virtual_dev_id), memory_limit(child_size), stream(stream)
从父内存池中分配内存给子内存池,设置成员变量,与流句柄

MemoryAllocator::~MemoryAllocator()
如果是子内存池,改动父内存的配置属性返回即可
如果是父内存池,调用cudaFree释放池化内存

void MemoryAllocator::freeze_model_memory()
整理当前内存池中的内存块,将内存块列表迁移给冻结内存块列表


Memory MemoryAllocator::alloc(size_t num_bytes, size_t round_up_bytes)
申请指定大小内存块,从池化内存中分配一段地址得到内存块,插入内存块列表,整理内存碎片

void MemoryAllocator::free(void* ptr, size_t size)
释放指定内存块,从内存块列表中移除,改动内存块的指向地址,整理内存碎片

DirectMemoryAllocator::DirectMemoryAllocator(
    int dev_id, int virtual_dev_id, size_t memory_limit, cudaStream_t stream)
    : MemoryAllocator(dev_id, virtual_dev_id, 1024, stream)
直接内存分配器,设置成员变量,与流句柄

Memory DirectMemoryAllocator::alloc(size_t size, size_t round_up_bytes)
直接分配内存,不进行整理内存碎片

void DirectMemoryAllocator::free(void* ptr, size_t size) 
直接释放内存,不进行整理内存碎片

void DirectMemoryAllocator::free_session()
从 frees列表里面cudafree所有块