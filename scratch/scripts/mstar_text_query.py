import asyncio, os, aiohttp
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
from benchmark.request import OurSystem, RequestInput
from benchmark.base import Qwen3Omni, RequestType
URL = "http://127.0.0.1:8103"
async def main():
    model = Qwen3Omni()
    ri = RequestInput(req_type=RequestType.T2T, prompt="How would the papers talk about it?")
    our = OurSystem()
    async with aiohttp.ClientSession() as s:
        m = await our.send_request(s, ri, URL, 0, model)
        txt = "".join(getattr(m, "_text_chunks", [])).replace("<|im_end|>", "")
        print(f"=== M* answer to the question posed as a TEXT QUERY ({len(txt)} chars) ===")
        print(txt)
asyncio.run(main())
