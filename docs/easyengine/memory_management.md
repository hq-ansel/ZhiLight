CUDA 的 **Memory Management** 是 CUDA 编程中非常重要的一部分，主要涉及主机（Host）和设备（Device）之间的内存分配、拷贝和释放。以下是 CUDA 内存管理的主要 API 总结，包括函数签名、使用示例和适用场景。

---

### 1. **设备内存管理**
设备内存是 GPU 上的内存，用于存储 CUDA 核函数需要处理的数据。

#### **API 1: `cudaMalloc`**
• **函数签名**：
  ```cpp
  cudaError_t cudaMalloc(void** devPtr, size_t size);
  ```
• **功能**：在设备上分配指定大小的内存。
• **参数**：
  • `devPtr`：指向设备内存指针的指针。
  • `size`：要分配的内存大小（以字节为单位）。
• **返回值**：成功返回 `cudaSuccess`，否则返回错误代码。
• **示例**：
  ```cpp
  float* d_data;
  size_t size = 1024 * sizeof(float);
  cudaMalloc((void**)&d_data, size);
  ```
• **场景**：在 GPU 上分配内存，用于存储 CUDA 核函数需要处理的数据。

#### **API 2: `cudaFree`**
• **函数签名**：
  ```cpp
  cudaError_t cudaFree(void* devPtr);
  ```
• **功能**：释放设备上分配的内存。
• **参数**：
  • `devPtr`：指向设备内存的指针。
• **返回值**：成功返回 `cudaSuccess`，否则返回错误代码。
• **示例**：
  ```cpp
  cudaFree(d_data);
  ```
• **场景**：释放不再需要的设备内存，避免内存泄漏。

---

### 2. **主机与设备之间的内存拷贝**
用于在主机和设备之间传输数据。

#### **API 3: `cudaMemcpy`**
• **函数签名**：
  ```cpp
  cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
  ```
• **功能**：在主机和设备之间拷贝数据。
• **参数**：
  • `dst`：目标内存指针。
  • `src`：源内存指针。
  • `count`：要拷贝的字节数。
  • `kind`：拷贝方向，可以是以下值：
    ◦ `cudaMemcpyHostToHost`：主机到主机。
    ◦ `cudaMemcpyHostToDevice`：主机到设备。
    ◦ `cudaMemcpyDeviceToHost`：设备到主机。
    ◦ `cudaMemcpyDeviceToDevice`：设备到设备。
• **返回值**：成功返回 `cudaSuccess`，否则返回错误代码。
• **示例**：
  ```cpp
  float h_data[1024];
  float* d_data;
  cudaMalloc((void**)&d_data, sizeof(h_data));
  cudaMemcpy(d_data, h_data, sizeof(h_data), cudaMemcpyHostToDevice);
  ```
• **场景**：将主机数据拷贝到设备，或将设备结果拷贝回主机。

---

### 3. **固定内存（Pinned Memory）**
固定内存是主机内存的一种特殊形式，可以提高主机与设备之间的数据传输效率。

#### **API 4: `cudaHostAlloc`**
• **函数签名**：
  ```cpp
  cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags);
  ```
• **功能**：在主机上分配固定内存。
• **参数**：
  • `pHost`：指向主机内存指针的指针。
  • `size`：要分配的内存大小（以字节为单位）。
  • `flags`：分配标志，常用的有 `cudaHostAllocDefault` 或 `cudaHostAllocMapped`。
• **返回值**：成功返回 `cudaSuccess`，否则返回错误代码。
• **示例**：
  ```cpp
  float* h_data;
  cudaHostAlloc(&h_data, 1024 * sizeof(float), cudaHostAllocDefault);
  ```
• **场景**：需要频繁在主机和设备之间传输数据时，使用固定内存可以提高性能。

#### **API 5: `cudaFreeHost`**
• **函数签名**：
  ```cpp
  cudaError_t cudaFreeHost(void* ptr);
  ```
• **功能**：释放固定内存。
• **参数**：
  • `ptr`：指向固定内存的指针。
• **返回值**：成功返回 `cudaSuccess`，否则返回错误代码。
• **示例**：
  ```cpp
  cudaFreeHost(h_data);
  ```
• **场景**：释放不再需要的固定内存。

---

### 4. **零拷贝内存（Zero-Copy Memory）**
零拷贝内存允许主机和设备直接共享同一块内存，避免显式的数据传输。

#### **API 6: `cudaHostAlloc`（映射到设备）**
• **函数签名**：
  ```cpp
  cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags);
  ```
• **功能**：在主机上分配固定内存，并映射到设备地址空间。
• **参数**：
  • `flags`：设置为 `cudaHostAllocMapped`。
• **返回值**：成功返回 `cudaSuccess`，否则返回错误代码。
• **示例**：
  ```cpp
  float* h_data;
  cudaHostAlloc(&h_data, 1024 * sizeof(float), cudaHostAllocMapped);
  ```
• **场景**：需要在主机和设备之间共享数据时，避免显式的数据传输。

#### **API 7: `cudaHostGetDevicePointer`**
• **函数签名**：
  ```cpp
  cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags);
  ```
• **功能**：获取主机固定内存的设备指针。
• **参数**：
  • `pDevice`：指向设备指针的指针。
  • `pHost`：主机固定内存指针。
  • `flags`：保留参数，通常为 0。
• **返回值**：成功返回 `cudaSuccess`，否则返回错误代码。
• **示例**：
  ```cpp
  float* d_data;
  cudaHostGetDevicePointer(&d_data, h_data, 0);
  ```
• **场景**：在零拷贝内存模式下，获取主机固定内存的设备指针。

---

### 5. **统一内存（Unified Memory）**
统一内存简化了主机和设备之间的内存管理，由 CUDA 运行时自动管理数据迁移。

#### **API 8: `cudaMallocManaged`**
• **函数签名**：
  ```cpp
  cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal);
  ```
• **功能**：分配统一内存。
• **参数**：
  • `devPtr`：指向统一内存指针的指针。
  • `size`：要分配的内存大小（以字节为单位）。
  • `flags`：内存附加标志，通常为 `cudaMemAttachGlobal`。
• **返回值**：成功返回 `cudaSuccess`，否则返回错误代码。
• **示例**：
  ```cpp
  float* u_data;
  cudaMallocManaged(&u_data, 1024 * sizeof(float));
  ```
• **场景**：简化主机和设备之间的内存管理，适合复杂的内存访问模式。

#### **API 9: `cudaFree`**
• **功能**：释放统一内存。
• **示例**：
  ```cpp
  cudaFree(u_data);
  ```
• **场景**：释放不再需要的统一内存。

---

### 总结
| **API**               | **功能**                     | **场景**                                   |
|-----------------------|----------------------------|------------------------------------------|
| `cudaMalloc`          | 分配设备内存                   | 在 GPU 上分配内存，用于核函数处理数据。            |
| `cudaFree`            | 释放设备内存                   | 释放 GPU 内存，避免内存泄漏。                  |
| `cudaMemcpy`          | 主机与设备之间的内存拷贝         | 数据传输。                                   |
| `cudaHostAlloc`       | 分配固定内存                   | 提高主机与设备之间的数据传输效率。               |
| `cudaFreeHost`        | 释放固定内存                   | 释放固定内存。                                |
| `cudaMallocManaged`   | 分配统一内存                   | 简化主机和设备之间的内存管理。                  |
| `cudaHostGetDevicePointer` | 获取主机固定内存的设备指针 | 零拷贝内存模式下，共享主机和设备内存。            |


### 附录1，为什么内存申请使用void**指针？
```
int* dptr= nullptr;
size_t bytes = sizeof(int)*10;
cudaError_t cuda_error = cudaMalloc((void**)&d_indata,bytes);
```

上面示例`cudaMalloc`的函数模型为`cudaError_t cudaMalloc (void **devPtr, size_t size ); `作用是在GPU上分配内存，分配线性大小的内存，devPtr是返回一个指向已经分配内存的指针，也就是CPU的devPtr所在的内存单元存的是GPU分配的显存首地址。

问题：`cudaMalloc()`为什么用`void**`而不是`void*`？

dptr这个指针是存储在CPU的内存上，假设dptr= 0x001f，int** ddptr = &dptr，ddptr = 0x034f。

假设使用void*，那么直接传入dptr，cudaMalloc函数在传入dptr时，先进行形参devPtr复制一份dptr这步操作，然后GPU再分配内存，实际上改变的是局部变量devPtr的值。等同于下面代码：
```
void* devptr = dptr;
devptr = malloc(10);
```
GPU分配内存的首地址写入了devPtr内存单元。那么cudaMalloc执行完后，dptr内存单元的数值没有改变，依旧为0x001f，通过dptr是访问不到GPU分配的内存。

讲那么多废话，其实就是一句话，传入参数都是形参，修改值就需要传入地址，修改地址就需要传入二级指针，只有C的实现，没有C++的引用那些东西。

使用void**就可以避免这个问题。传入的是dptr的指针ddptr，在cudaMalloc函数内部会执行，那么*ddptr就是dptr的地址，直接存储GPU分配内存的首地址。
```
*ddptr= malloc(10);
```