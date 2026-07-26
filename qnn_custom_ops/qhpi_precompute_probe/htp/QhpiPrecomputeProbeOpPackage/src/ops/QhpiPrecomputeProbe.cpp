#include "HTP/core/constraints.h"
#include "HTP/core/qhpi.h"
#include <cstdint>

namespace {

constexpr uint32_t kMagic = 0x50524543U;

struct ProbeData {
  uint32_t magic;
  uint32_t precompute_calls;
  uint32_t reserved0;
  uint32_t reserved1;
};

static uint32_t precompute(QHPI_RuntimeHandle *,
                           void *data,
                           uint32_t,
                           QHPI_Tensor **,
                           uint32_t,
                           const QHPI_Tensor *const *) {
  if (data == nullptr) {
    return QHPI_ErrorFatal;
  }
  auto *probe = static_cast<ProbeData *>(data);
  probe->magic = kMagic;
  probe->precompute_calls += 1;
  return QHPI_Success;
}

static uint32_t execute(QHPI_RuntimeHandle *, const void *data) {
  if (data == nullptr) {
    return QHPI_ErrorFatal;
  }
  const auto *probe = static_cast<const ProbeData *>(data);
  return probe->magic == kMagic && probe->precompute_calls == 1
             ? QHPI_Success
             : QHPI_ErrorFatal;
}

static uint32_t executeBaseline(QHPI_RuntimeHandle *,
                                uint32_t,
                                QHPI_Tensor **,
                                uint32_t,
                                const QHPI_Tensor *const *) {
  return QHPI_Success;
}

static float cost(uint32_t, const QHPI_Tensor *const *) {
  return 1.0f;
}

static const QHPI_Op *earlyRewrite(const QHPI_Op *op) {
  return op;
}

static QHPI_Shape shapeRequired(const QHPI_Op *) {
  return QHPI_Shape{0};
}

static QHPI_Shape shapeLegal(const QHPI_Op *, const QHPI_Shape *shape) {
  return *shape;
}

static const QHPI_Op *buildTile(const QHPI_Op *op,
                                const QHPI_Shape *,
                                const QHPI_Shape *) {
  return op;
}

static const QHPI_Op *lateRewrite(const QHPI_Op *op) {
  return op;
}

static QHPI_Tensor_Signature_v1 inputs[] = {
    {
        .element_type = QHPI_Float16,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM,
    },
    {
        .element_type = QHPI_Float16,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM,
    },
};

static QHPI_Tensor_Signature_v1 outputs[] = {
    {
        .element_type = QHPI_Float32,
        .layout = QHPI_Layout_Flat4,
        .storage = QHPI_Storage_Direct,
        .mem_placement = QHPI_MemLoc_DDR_OR_TCM,
    },
};

static QHPI_Kernel_v1 kernel = {
    .function_name = "qhpi_precompute_probe_execute",
    .function = nullptr,
    .resources = QHPI_RESOURCE_MAIN,
    .source_destructive = false,
    .multithreaded = false,
    .variable_inputs = false,
    .variable_outputs = false,
    .min_inputs = 2,
    .input_signature = inputs,
    .min_outputs = 1,
    .output_signature = outputs,
    .cost_function = cost,
    .sync_block_size = 0,
    .precomputed_data_size = sizeof(ProbeData),
    .do_precomputation_function = precompute,
    .function_with_precomputed_data = execute,
    .predicate = nullptr,
};

static QHPI_Kernel_v1 kernels[] = {kernel};

}  // namespace

QHPI_OpInfo_v1 qhpiprecomputeprobeOpInfo = {
    .name = THIS_PKG_NAME_STR "::QhpiPrecomputeProbe",
    .num_kernels = 1,
    .kernels = kernels,
    .early_rewrite = earlyRewrite,
    .shape_required = shapeRequired,
    .shape_legalized = shapeLegal,
    .build_tile = buildTile,
    .late_rewrite = lateRewrite,
};

static QHPI_OpInfo_v1 ops[] = {qhpiprecomputeprobeOpInfo};

extern "C" void register_qhpiprecomputeprobe_ops() {
  qhpi_register_ops_v1(sizeof(ops) / sizeof(ops[0]), ops,
                       THIS_PKG_NAME_STR);
}
