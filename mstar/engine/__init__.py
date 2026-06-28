import torch

from mstar.engine.precision import apply_matmul_precision, log_precision_settings

torch._dynamo.config.recompile_limit = 84
torch._dynamo.config.allow_unspec_int_on_nn_module = True
torch._dynamo.config.specialize_int = False

# fp32 matmul precision: env-gated (MSTAR_FP32_MATMUL_PRECISION), default 'high'
# preserves the previously hardcoded behavior. On Hopper, 'high'/'medium' route
# fp32 matmuls through TF32 tensor cores; 'highest' keeps true IEEE fp32.
_matmul_precision = apply_matmul_precision()
log_precision_settings(_matmul_precision)
