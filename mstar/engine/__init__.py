import torch

# 84 covered the default capture grids; the eiv2 env-widened prefill grids
# (MSTAR_PREFILL_BATCH_SIZES x MSTAR_PREFILL_BUCKETS) push distinct compile
# shapes past it, and overflow silently degrades small-bucket kernels to
# eager-quality. Headroom is free (limit only bounds cache size).
torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.allow_unspec_int_on_nn_module = True
torch._dynamo.config.specialize_int = False
torch.set_float32_matmul_precision('high')
