# learn from ZhiLight

my model dir

/home/ubuntu/data/exp/proj2410/model

code quality

[Google Python style guide](https://google.github.io/styleguide/pyguide.html) and [Google C++ style guide](https://google.github.io/styleguide/cppguide.html).

## step 1: try build

but not install cmake
first install cmake

download  cmake release version 4.0.0-rc3

```bash
wget https://github.com/Kitware/CMake/releases/download/v4.0.0-rc3/cmake-4.0.0-rc3.tar.gz
```

```bash
tar -zxvf cmake-4.0.0-rc3.tar.gz
```

in cmake-4.0.0-rc3/

prerequisite install openssl

```bash
sudo apt install libssl-dev
run ./bootstrap 
```

this operation will generate a configure file for building cmake

then run compiile

```bash
make 

make install

make &&  make install
```

```bash
prerequisite install nccl first
sudo dpkg -i nccl-local-repo-ubuntu2004-2.25.1-cuda12.4_1.0-1_amd64.deb 
sudo cp /var/nccl-local-repo-ubuntu2004-2.25.1-cuda12.4/nccl-local-A7470B26-keyring.gpg /usr/share/keyrings/
 sudo apt install libnccl2=2.25.1-1+cuda12.4 libnccl-dev=2.25.1-1+cuda12.4

CMAKE_BUILD_PARALLEL_LEVEL=32 TESTING=0 python setup.py bdist_wheel
```

```
python setup.py bdist_wheel
```

* 这是 Python 的标准构建命令，用于将 Python 项目打包成 `.whl` 文件（Wheel 格式的二进制分发包）。

$ CMAKE_GENERATER="Ninja" python setup.py install

cd ./ZhiLight && pip install -e .

python -m zhilight.server.openai.entrypoints.api_server --model-path /home/ubuntu/data/exp/proj2410/model/Qwen2.5-14B --dyn-max-batch-size 4

nsys profile -o output_file_name -- python -m zhilight.server.openai.entrypoints.api_server --model-path /home/ubuntu/data/exp/proj2410/model/Qwen2.5-14B -dyn-max-batch-size 4

计划重写内存分配部分

使用nvml库管理显存,这个库通常安装cudatoolkit的时候会顺便装上




问题回顾
在编译 C++ 程序时，尝试使用 #include <nvml.h> 并链接 NVML 库，但遇到了以下错误：

/usr/bin/ld: 找不到 -lnvidia-ml
collect2: error: ld returned 1 exit status
通过查找，发现系统中存在 libnvidia-ml.so.1，但链接器默认查找的是 libnvidia-ml.so，因此需要手动创建符号链接。

​解决步骤
​1. 确认 libnvidia-ml.so 的路径
运行以下命令，确认 libnvidia-ml.so 的位置：

bash
find / -name libnvidia-ml*
输出示例：

/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.550.107.02
可以看到，libnvidia-ml.so.1 是一个符号链接，指向 libnvidia-ml.so.550.107.02，但缺少 libnvidia-ml.so。

​2. 创建符号链接
由于链接器默认查找 libnvidia-ml.so，需要手动创建一个符号链接，指向 libnvidia-ml.so.1：

bash
sudo ln -s /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so
运行以下命令，确认符号链接已创建：

bash
ls -l /usr/lib/x86_64-linux-gnu/libnvidia-ml.so*
输出示例：

lrwxrwxrwx 1 root root 26 Sep 20 19:59 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so -> libnvidia-ml.so.1
lrwxrwxrwx 1 root root 26 Sep 20 19:59 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 -> libnvidia-ml.so.550.107.02
-rwxr-xr-x 1 root root 2078360 Sep 19 15:43 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.550.107.02
​3. 编译程序
使用以下命令编译程序：

bash
g++ -o test_nvml test_nvml.cpp -I/usr/local/cuda-12.4/targets/x86_64-linux/include -L/usr/lib/x86_64-linux-gnu -lnvidia-ml
-I/usr/local/cuda-12.4/targets/x86_64-linux/include：指定 nvml.h 头文件的路径。
-L/usr/lib/x86_64-linux-gnu：指定 libnvidia-ml.so 库文件的路径。
-lnvidia-ml：链接 libnvidia-ml.so 库。
​4. 运行程序
运行编译后的程序：

bash
./test_nvml
如果 NVML 初始化成功，并且系统中有 GPU，输出可能类似于：

Found 1 GPU(s)



TODO:

- [ ] 重写内存分配部分
- [ ] 先把库删减完成
- [ ] 再写一下CUDA的各个核测试代码