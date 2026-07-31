"""Capture vLLM's actual A2T text via streaming SSE (parsing delta.content)."""
import base64, json, requests

SYS = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
       "capable of perceiving auditory and visual inputs, as well as generating text and speech.")
PROMPT = "Please provide a transcription of the spoken content."
URL = "http://127.0.0.1:8093/v1/chat/completions"

for i in range(5):
    wav = f"/home/tim/tmp/libri_wavs/librispeech_{i:05d}.wav"
    try:
        b64 = base64.b64encode(open(wav, "rb").read()).decode()
    except FileNotFoundError:
        print(f"clip {i}: wav missing"); continue
    payload = {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYS}]},
            {"role": "user", "content": [
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64}"}},
                {"type": "text", "text": PROMPT}]},
        ],
        "temperature": 0.0, "max_tokens": 256, "seed": 42,
        "modalities": ["text"], "stream": True, "stream_options": {"include_usage": True},
    }
    text, ntok, finish = "", 0, None
    try:
        r = requests.post(URL, json=payload, headers={"Authorization": "Bearer EMPTY"},
                          stream=True, timeout=180)
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8", "replace")
            if not line.startswith("data: "):
                continue
            d = line[6:]
            if d.strip() == "[DONE]":
                break
            obj = json.loads(d)
            for ch in obj.get("choices", []):
                c = (ch.get("delta") or {}).get("content")
                if c:
                    text += c; ntok += 1
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
        print(f"\n===== clip {i} =====  finish={finish} text_chunks={ntok} chars={len(text)}")
        print(f"   {text!r}")
    except Exception as e:
        print(f"clip {i}: ERROR {e}")
