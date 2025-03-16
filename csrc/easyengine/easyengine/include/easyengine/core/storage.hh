#pragma once
#include "easyengine/core/tensor.hh"
#include "easyengine/core/export.hh"
#include <string>
#include <vector>

namespace easyengine {

namespace core {

enum class DataType;

class ENGINE_EXPORT ParameterData {
public:
    std::string name;
    std::vector<size_t> shape;
    DataType dtype;
    char* ptr;
    size_t nbytes;
    bool own { true };

    ParameterData(const std::string& name, const std::vector<size_t>& shape, DataType dtype);
    ~ParameterData();
    ParameterData(ParameterData&&);
    ParameterData(const ParameterData&) = delete;
};

class ENGINE_EXPORT Storage {
public:
    Storage();
    virtual ~Storage();
    Storage(const Storage&) = delete;
    Storage(Storage&&) = delete;
    virtual void fetch_parameter(ParameterData& data);
    virtual void fetch_parameter(const std::string& name, Tensor& data);
    virtual size_t used_memory() const;
};

}

}