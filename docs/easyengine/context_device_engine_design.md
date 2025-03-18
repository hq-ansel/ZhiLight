


关键函数
ContextImpl::ContextImpl(EngineImpl* engine, const std::vector<int>& devices, int rank, bool aux)
    : engine(engine),
      active_device(-1),
      devices(devices),
      rank_(rank),
      used_memory(0),
      peak_memory(0),
      thread_id(std::this_thread::get_id()),
      debug(0),
      tensor_id(0L),
      aux_(aux) 
读取 engineImpl ，将tensor_cache这一map<device_id,tensor_ptr> 调整大小为8？这个magic number应该是对应一个节点有八个GPU
遍历devices 把engine中的device handles（DeviceHandles* {}） 加入dev_handles


大概层次 从高到低
Context(ContextImpl)
Tensor(TensorImpl)
DeviceHandles MemoryAllocator
GPUInfo Stream
DeviceGuard DeviceHandles DeviceConfiguration
EngineException
Memory_ 
ncclComm_t 
cublasHandle_t
cudaEvent_t 
cudaStream_t
cudaDeviceProp


