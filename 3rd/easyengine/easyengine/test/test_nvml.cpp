#include <nvml.h>
#include <iostream>

int main() {
    // 初始化 NVML
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        return 1;
    }

    // 获取 GPU 数量
    unsigned int deviceCount;
    result = nvmlDeviceGetCount(&deviceCount);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get device count: " << nvmlErrorString(result) << std::endl;
        nvmlShutdown();
        return 1;
    }

    std::cout << "Found " << deviceCount << " GPU(s)" << std::endl;

    // 关闭 NVML
    nvmlShutdown();
    return 0;
}