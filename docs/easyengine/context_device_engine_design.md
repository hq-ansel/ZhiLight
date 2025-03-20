


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


Tensor Context::tensor(
        const std::vector<size_t>& shape,
        DataType dtype,
        const std::string& name,
        size_t round_up_bytes) 
用法 ctx.tensor(shape, dtype) name==emptyStr round_up_bytes== 1024
常见的用法是 auto c = ctx.tensor(c_shape, data_type, "", round_up_n ) 或者采用
auto out = ctx.tensor(out_shape, A.dtype()) 进行tensor创建



Tensor Context::parameter(const std::vector<size_t>& shape, DataType dtype) 
用法 ctx.parameter({ dim_out, dim_in }, core::DataType::kInt8)
具体的实现上甚至和tensor很相似，区别在于实现tensor的时候隐式的调用alloc，而parameter得到一个空指针，仅仅有预计大小
即空有形状，没有地址，但是绑定了设备


Tensor Context::distribute_parameter(const Tensor& param, DistLayout layout) 
一个尝试把所有的tensor分配到不同的gpu，没有案例实现，对于如果采用非主行的Layout!=DistLayout::ROW
会把`shape[:,shard_len,:]`的shard_len 按照cdiv切分划归到不同的rank上，


void Context::load_parameter(
    Tensor* weight,
    const std::string& name,
    const std::map<std::string, const Tensor>& state_dict,
    bool parallel,
    DistLayout layout)
从state_dict中加载参数，并分配到对应的设备上，如果parallel==false,直接调用assign_or_copy API进行复制
如果parallel==true，则会采用nccl进行通信


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


