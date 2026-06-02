"""Chat with a running mminf server via the Python SDK.

Start a server first, e.g.:  mminf serve bagel
"""

from mminf import MMInfClient
from mminf.client import TextChunk

client = MMInfClient("http://localhost:8000")

# Non-streaming
print(client.chat("What is the capital of France?").text)

# Streaming (yields TextChunk / ImageChunk / AudioChunk)
for event in client.chat("Tell me a short story.", stream=True):
    if isinstance(event, TextChunk):
        print(event.text, end="", flush=True)
print()
