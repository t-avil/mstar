import sys; sys.modules.setdefault("flash_attn", None)  # no flash-attn; SDPA both
import time, torch, json
DEV="cuda:0"; DT=torch.bfloat16
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoderConfig, Qwen3OmniMoeAudioEncoderConfig)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoder, Qwen3OmniMoeAudioEncoder)
from mstar.model.qwen3_omni.components.vision_encoder import NativeQwen3OmniVisionEncoder
from mstar.model.qwen3_omni.components.audio_encoder import NativeQwen3OmniAudioEncoder
def t(fn,n=5,w=2):
    for _ in range(w): fn()
    torch.cuda.synchronize(); s=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-s)/n*1000
res={}
# VISION (one 728-patch image, the I2T encoder)
vc=Qwen3OmniMoeVisionEncoderConfig()
hf=Qwen3OmniMoeVisionEncoder._from_config(vc,attn_implementation="sdpa").to(DEV,DT).eval()
nat=NativeQwen3OmniVisionEncoder(vc).to(DEV,DT).eval()
rows=vc.in_channels*vc.temporal_patch_size*vc.patch_size*vc.patch_size
g=torch.tensor([[1,26,28]],device=DEV); pv=torch.randn(26*28,rows,device=DEV,dtype=DT)
with torch.no_grad():
    res['vision_before_HF_ms']=round(t(lambda:hf(pv,grid_thw=g),2,1),1)
    res['vision_after_native_ms']=round(t(lambda:nat(pv,grid_thw=g)),2)
# AUDIO (~30s clip, the S2T encoder)
ac=Qwen3OmniMoeAudioEncoderConfig(); ac.n_window=50; ac.n_window_infer=800
hfa=Qwen3OmniMoeAudioEncoder._from_config(ac,attn_implementation="sdpa").to(DEV,DT).eval()
nata=NativeQwen3OmniAudioEncoder(ac).to(DEV,DT).eval()
lens=torch.tensor([3000],device=DEV); f=torch.randn(ac.num_mel_bins,3000,device=DEV,dtype=DT)
with torch.no_grad():
    res['audio_before_HF_ms']=round(t(lambda:hfa(f,feature_lens=lens)),2)
    res['audio_after_native_ms']=round(t(lambda:nata(f,lens)),2)
res['vision_speedup']=round(res['vision_before_HF_ms']/res['vision_after_native_ms'],1)
res['audio_speedup']=round(res['audio_before_HF_ms']/res['audio_after_native_ms'],2)
print(json.dumps(res,indent=2))
json.dump(res,open("/home/timchick/mstar/benchmark/artifacts/encoder_before_after.json","w"),indent=2)
