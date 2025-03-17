#pragma once
#include "easyengine/core/core.hh"

namespace easyengine {
namespace functions {

core::Tensor typecast(const core::Context& ctx, const core::Tensor& in, core::DataType out_type);

}
}