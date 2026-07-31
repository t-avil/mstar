"""Dump Thinker text: M*-new (A2T) vs vLLM (A2S — its working path; the harness
parser buffers the thinker text deltas into _text_chunks even in S2S mode)."""
import asyncio, os, aiohttp
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
from benchmark.request import OurSystem, VLLMOmni
from benchmark.base import Qwen3Omni, RequestType
from benchmark.dataset import LibriSpeechDataset

OUR_URL = "http://127.0.0.1:8094"
VLLM_URL = "http://127.0.0.1:8093"

async def main():
    model = Qwen3Omni()
    ds = LibriSpeechDataset(num_requests=5, req_type=RequestType.A2T,
                            local_file_dir="/home/tim/tmp/libri_wavs")
    items = ds.items[:5]
    our, vll = OurSystem(), VLLMOmni()
    async with aiohttp.ClientSession() as s:
        for i, ri in enumerate(items):
            ri.req_type = RequestType.A2T
            mo = await our.send_request(s, ri, OUR_URL, i, model)
            ri.req_type = RequestType.A2S  # vLLM's stable path; thinker text still streams
            mv = await vll.send_request(s, ri, VLLM_URL, i, model)
            to = "".join(getattr(mo, "_text_chunks", [])) or f"<err {getattr(mo,'error',None)}>"
            tv = "".join(getattr(mv, "_text_chunks", [])) or f"<err {getattr(mv,'error',None)}>"
            ct_v = getattr(mv, "output_text_tokens", None)
            print(f"\n===== clip {i} =====")
            print(f"M*-new : {to!r}")
            print(f"vLLM   (thinker_toks={ct_v}): {tv!r}")

asyncio.run(main())
