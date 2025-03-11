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