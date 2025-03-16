/*
* DataType 用来定义有哪些数据类型可用 都是名称和枚举类型
*/
#pragma once
#include "easyengine/core/export.hh"
#include <memory>
#include <initializer_list>
#include <string>
#include <vector>
#include <stdexcept>

namespace easyengine {
namespace core {
/**
* @enum DataType
* @brief 定义支持的数据类型枚举
*/
enum class DataType {
    kDouble,
    kFloat,
    kHalf,
    kInt8,
    kInt16,
    kInt32,
    kBFloat16,
    kFP8_E4M3,
    kFP8_E5M2,
};

ENGINE_EXPORT const char* get_data_type_name(DataType dtype);
ENGINE_EXPORT DataType name_to_data_type(const std::string& name);

template<typename T>
struct DTypeDeducer {
    static DataType data_type() { throw std::runtime_error("data_type must be overwrite"); }
};

template<>
struct DTypeDeducer<int> {
    static DataType data_type() { return DataType::kInt32; }
};
template<>
struct DTypeDeducer<int8_t> {
    static DataType data_type() { return DataType::kInt8; }
};
template<>
struct DTypeDeducer<float> {
    static DataType data_type() { return DataType::kFloat; }
};
template<>
struct DTypeDeducer<void*> {
    static DataType data_type() { return DataType::kDouble; }
};

#define EZENGINE_PRIVATE_DTYPE_DISPATCH_CASE(dtypename, realname, ...)                                   \
    case easyengine::core::DataType::dtypename: {                                                    \
        using scalar_t = realname;                                                                 \
        { __VA_ARGS__ }                                                                            \
    } break;
// do while(0) 代码块展开小技巧，用于在宏中执行多条语句，并最后返回最后一条语句的结果
// 不会被误认为是语句块的结束，而是作为一个整体被执行。
#define EZENGINE_DTYPE_DISPATCH(dtype, ...)                                                              \
    do {                                                                                           \
        const auto& _dtype = dtype;                                                                \
        switch (_dtype) {                                                                          \
            EZENGINE_PRIVATE_DTYPE_DISPATCH_CASE(kDouble, double, __VA_ARGS__)                           \
            EZENGINE_PRIVATE_DTYPE_DISPATCH_CASE(kFloat, float, __VA_ARGS__)                             \
            EZENGINE_PRIVATE_DTYPE_DISPATCH_CASE(kHalf, half, __VA_ARGS__)                               \
            EZENGINE_PRIVATE_DTYPE_DISPATCH_CASE(kInt8, int8_t, __VA_ARGS__)                             \
            EZENGINE_PRIVATE_DTYPE_DISPATCH_CASE(kInt16, int16_t, __VA_ARGS__)                           \
            EZENGINE_PRIVATE_DTYPE_DISPATCH_CASE(kInt32, int32_t, __VA_ARGS__)                           \
            EZENGINE_PRIVATE_DTYPE_DISPATCH_CASE(kBFloat16, nv_bfloat16, __VA_ARGS__)                    \
            default: BM_EXCEPTION("Unsupported data type");                                        \
        };                                                                                         \
    } while (0)

#define EZENGINE_DTYPE_DISPATCH_FLOAT(dtype, ...)                                                        \
    do {                                                                                           \
        const auto& _dtype = dtype;                                                                \
        switch (_dtype) {                                                                          \
            EZENGINE_PRIVATE_DTYPE_DISPATCH_CASE(kFloat, float, __VA_ARGS__)                             \
            EZENGINE_PRIVATE_DTYPE_DISPATCH_CASE(kHalf, half, __VA_ARGS__)                               \
            EZENGINE_PRIVATE_DTYPE_DISPATCH_CASE(kBFloat16, nv_bfloat16, __VA_ARGS__)                    \
            default: BM_EXCEPTION("Unsupported data type");                                        \
        };                                                                                         \
    } while (0)

#define EZENGINE_DTYPE_DISPATCH_HALF(dtype, ...)                                                         \
    do {                                                                                           \
        const auto& _dtype = dtype;                                                                \
        switch (_dtype) {                                                                          \
            EZENGINE_PRIVATE_DTYPE_DISPATCH_CASE(kHalf, half, __VA_ARGS__)                               \
            EZENGINE_PRIVATE_DTYPE_DISPATCH_CASE(kBFloat16, nv_bfloat16, __VA_ARGS__)                    \
            default: BM_EXCEPTION("data type is not half or BF16");                                \
        };                                                                                         \
    } while (0)

} // namespace core
} // namespace easyengine

namespace std {
static inline std::string to_string(easyengine::core::DataType dt) {
    return easyengine::core::get_data_type_name(dt);
}
}
