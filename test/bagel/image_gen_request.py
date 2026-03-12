#!/usr/bin/env python3

import requests
import base64
import json
import sys


URL = "http://0.0.0.0:8000/generate"


def main():
    with requests.post(
        URL,
        data={
            "text": "a car made out of small cars",
            # "text": "A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere.",
            "output_modalities": "image",
        },
        stream=True,
    ) as resp:
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            modality = msg.get("modality")
            data_b64 = msg.get("data", "")

            if not data_b64:
                continue

            decoded = base64.b64decode(data_b64)

            if modality == "text":
                sys.stdout.write(decoded.decode("utf-8", errors="replace"))
                sys.stdout.flush()

            elif modality == "image":
                with open("output.png", "wb") as f:
                    f.write(decoded)
                print("\nSaved image to output.png")


if __name__ == "__main__":
    main()