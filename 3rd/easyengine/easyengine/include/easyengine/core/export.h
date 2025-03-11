// 确保此文件在单次编译过程中只被包含一次，避免重复定义的问题。
#pragma once

// 定义ENGINE_EXPORT宏，使用GCC或Clang的特定属性来设置函数或类的可见性。
// 在创建共享库（动态链接库）时，这个宏通常用于导出公共API。
// "default"的可见性意味着这些符号将会被导出，并且可以从其他共享库或主程序中访问。
#define ENGINE_EXPORT __attribute__((visibility("default")))


/*
#define ENGINE_EXPORT __attribute__((visibility("default"))) 是 GCC 和 Clang 编译器提供的一种机制，用于控制符号（如函数、变量等）的可见性。以下是对 __attribute__ 和 visibility 的详细解释：

​1. __attribute__ 是什么？
__attribute__ 是 GCC 和 Clang 编译器提供的一种扩展语法，用于向编译器传递额外的信息或指示。它通常用于优化、调试或控制代码的行为。

​语法
__attribute__ 的语法如下：

cpp
__attribute__((attribute_name(parameters)))
其中：

attribute_name 是具体的属性名称。
parameters 是传递给属性的参数（如果有）。
​常见用途
__attribute__ 可以用于多种场景，例如：

控制符号的可见性（visibility）。
指定对齐方式（aligned）。
标记函数为不返回（noreturn）。
标记函数为弱符号（weak）。
​2. visibility 是什么？
visibility 是 __attribute__ 的一个具体属性，用于控制符号的可见性。它主要用于动态链接库（shared library）中，决定哪些符号（函数、变量等）可以被外部访问。

​可见性类型
visibility 支持以下几种模式：

​**default**：符号是可见的，可以被外部访问。这是默认行为。
​**hidden**：符号是不可见的，只能在当前库内部访问。
​**protected**：符号是可见的，但不能被覆盖（override）。
​**internal**：符号是内部使用的，不能被外部访问。
​用途
通过控制符号的可见性，可以实现以下目的：

减少动态链接库的符号表大小，提高加载性能。
防止外部代码直接访问库内部符号，增强封装性。
避免符号冲突（symbol collision）。


1. 编写动态链接库
​代码：engine.cpp
cpp
#include <iostream>

// 定义 ENGINE_EXPORT 宏
#define ENGINE_EXPORT __attribute__((visibility("default")))

// 公开函数，标记为 ENGINE_EXPORT
ENGINE_EXPORT void public_function() {
    std::cout << "This is a public function!" << std::endl;
}

// 私有函数，未标记 ENGINE_EXPORT
void private_function() {
    std::cout << "This is a private function!" << std::endl;
}
​编译动态链接库
使用以下命令将 engine.cpp 编译为动态链接库：

bash
g++ -fvisibility=hidden -shared -o libengine.so engine.cpp
-fvisibility=hidden：将未明确标记为 default 的符号设置为 hidden。
-shared：生成动态链接库。
-o libengine.so：指定输出文件名为 libengine.so。

*/