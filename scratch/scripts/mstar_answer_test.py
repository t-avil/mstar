"""Prove M* can 'behave like vLLM' — generate a long answer instead of a short
transcription — on clip4 (the spoken question 'How would the papers talk about it?').
Same audio, different user instruction. M* puts the audio as a bare block + a text
instruction turn, so the instruction governs."""
import asyncio, os, aiohttp
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
from benchmark.request import OurSystem
from benchmark.base import Qwen3Omni, RequestType
from benchmark.dataset import LibriSpeechDataset

URL = "http://127.0.0.1:8103"
PROMPTS = {
    "DEFAULT transcribe": "Please provide a transcription of the spoken content.",
    "answer the question": "Answer the question.",
    "respond to audio": "Respond in detail to what was said in the audio.",
}

async def main():
    model = Qwen3Omni()
    ds = LibriSpeechDataset(num_requests=5, req_type=RequestType.A2T,
                            local_file_dir="/home/tim/tmp/libri_wavs")
    clip4 = ds.items[4]  # "How would the papers talk about it?"
    our = OurSystem()
    async with aiohttp.ClientSession() as s:
        for label, p in PROMPTS.items():
            clip4.prompt = p
            clip4.req_type = RequestType.A2T
            m = await our.send_request(s, clip4, URL, 0, model)
            txt = "".join(getattr(m, "_text_chunks", [])).replace("<|im_end|>", "")
            print(f"\n===== [{label}]  prompt={p!r} =====")
            print(f"  M* output: {len(txt)} chars")
            print(f"  {txt[:600]!r}")

asyncio.run(main())
