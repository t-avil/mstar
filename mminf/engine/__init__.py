import torch
torch._dynamo.config.recompile_limit = 32
torch._dynamo.config.allow_unspec_int_on_nn_module = True
torch.set_float32_matmul_precision('high')