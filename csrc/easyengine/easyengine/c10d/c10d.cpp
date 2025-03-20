
#include "easyengine/c10d/c10d.hh"
#include "easyengine/core/tensor.hh"
#include "easyengine/core/exception.hh"

namespace easyengine {
namespace c10d {

ncclDataType_t dtype2nccl(core::DataType dtype) {
    switch (dtype) {
        case core::DataType::kInt8: return ncclInt8;
        case core::DataType::kDouble: return ncclDouble;
        case core::DataType::kFloat: return ncclFloat;
        case core::DataType::kHalf: return ncclHalf;
        case core::DataType::kBFloat16: return ncclBfloat16;
        case core::DataType::kInt32: return ncclInt32;
        default:
            EZ_ASSERT(false, "Unsupport dtype " + std::string(get_data_type_name(dtype)));
            return ncclNumTypes;
    }
}

void NCCLAllGather(const core::Context& ctx, const core::Tensor& sendbuff, core::Tensor& recvbuff) {
    EZ_NCCL_ASSERT(ncclAllGather(
        sendbuff.data<void*>(),
        recvbuff.mutable_data<void*>(),
        sendbuff.numel(),
        dtype2nccl(sendbuff.dtype()),
        ctx.current_comm(),
        ctx.current_stream()->ptr));
}

void NCCLCheckAsync(ncclComm_t comm) {
    auto state = ncclInProgress;
    do {
        ncclCommGetAsyncError(comm, &state);
    } while (state == ncclInProgress);
    EZ_NCCL_ASSERT(state);
}
/*
* NCCLAllReduce 函数：使用 NCCL（NVIDIA Collective Communications Library）执行 AllReduce 操作。
* AllReduce 是一种并行计算中常用的通信操作，它将所有参与进程的数据进行规约（如求和、最大值等），
* 并将结果广播回所有进程
* 读取send tensor的data数据指针写入到recv tensor的mutable_data数据指针，
* ncclAllReduce的用法 ncclAllReduce(sendbuff,recvbuff,numel,dtype,op,comm,stream)
*  rank[0]      rank[1]     rank[2]     rank[3]
*  0            1           2           3
*  after AllReduce:
*  rank[0]      rank[1]     rank[2]     rank[3]
*  op(0,1,2,3)  op(0,1,2,3) op(0,1,2,3) op(0,1,2,3)
*/
void NCCLAllReduce(
    const core::Context& ctx,
    const core::Tensor& sendbuff,
    core::Tensor& recvbuff,
    ncclRedOp_t op) {
    EZ_NCCL_ASSERT(ncclAllReduce(
        sendbuff.data<void*>(),
        recvbuff.mutable_data<void*>(),
        sendbuff.numel(),
        dtype2nccl(sendbuff.dtype()),
        op,
        ctx.current_comm(),
        ctx.current_stream()->ptr));
//    NCCLCheckAsync(ctx.current_comm());
}

void NCCLBroadcast(
    const core::Context& ctx, const core::Tensor& sendbuff, core::Tensor& recvbuff, int root) {
    EZ_NCCL_ASSERT(ncclBroadcast(
        sendbuff.data<void*>(),
        recvbuff.mutable_data<void*>(),
        sendbuff.numel(),
        dtype2nccl(sendbuff.dtype()),
        root,
        ctx.current_comm(),
        ctx.current_stream()->ptr));
}

void NCCLReduce(
    const core::Context& ctx,
    const core::Tensor& sendbuff,
    core::Tensor& recvbuff,
    ncclRedOp_t op,
    int root) {
    EZ_NCCL_ASSERT(ncclReduce(
        sendbuff.data<void*>(),
        recvbuff.mutable_data<void*>(),
        sendbuff.numel(),
        dtype2nccl(sendbuff.dtype()),
        op,
        root,
        ctx.current_comm(),
        ctx.current_stream()->ptr));
}

void NCCLReduceScatter(
    const core::Context& ctx,
    const core::Tensor& sendbuff,
    core::Tensor& recvbuff,
    ncclRedOp_t op) {
    EZ_NCCL_ASSERT(ncclReduceScatter(
        sendbuff.data<void*>(),
        recvbuff.mutable_data<void*>(),
        recvbuff.numel(),
        dtype2nccl(sendbuff.dtype()),
        op,
        ctx.current_comm(),
        ctx.current_stream()->ptr));
}

void NCCLSend(const core::Context& ctx, const core::Tensor& sendbuff, int peer) {
    EZ_NCCL_ASSERT(ncclSend(
        sendbuff.data<void*>(),
        sendbuff.numel(),
        dtype2nccl(sendbuff.dtype()),
        peer,
        ctx.current_comm(),
        ctx.current_stream()->ptr));
}

void NCCLRecv(const core::Context& ctx, core::Tensor& recvbuff, int peer) {
    EZ_NCCL_ASSERT(ncclRecv(
        recvbuff.mutable_data<void*>(),
        recvbuff.numel(),
        dtype2nccl(recvbuff.dtype()),
        peer,
        ctx.current_comm(),
        ctx.current_stream()->ptr));
}

void NCCLGroupStart() {
    EZ_NCCL_ASSERT(ncclGroupStart());
}

void NCCLGroupEnd() {
    EZ_NCCL_ASSERT(ncclGroupEnd());
}
void NCCLGroupEndCheck(ncclComm_t comm) {
    ncclResult_t ret = ncclGroupEnd();
    if (ret == ncclInProgress) {
        NCCLCheckAsync(comm);
    } else {
        EZ_NCCL_ASSERT(ret);
    }
}
int NCCLCommCount(const core::Context& ctx) {
    int res;
    EZ_NCCL_ASSERT(ncclCommCount(ctx.current_comm(), &res));
    return res;
}
int NCCLCommUserRank(const core::Context& ctx) {
    int rank;
    EZ_NCCL_ASSERT(ncclCommUserRank(ctx.current_comm(), &rank));
    return rank;
}
} // namespace c10d
} // namespace easyengine