#!/usr/bin/env python3

import requests
import base64
import json
import sys


URL = "http://0.0.0.0:8000/generate"


def main():
    with requests.post(
        URL,
        files={"text": (None, "Hello, how are you?")},
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

            if msg.get("modality") == "text":
                data_b64 = msg.get("data", "")
                decoded = base64.b64decode(data_b64)

                sys.stdout.write(decoded.decode("utf-8", errors="replace"))
                sys.stdout.flush()


if __name__ == "__main__":
    main()