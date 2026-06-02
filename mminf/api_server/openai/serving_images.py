"""/v1/images/generations handler (text-to-image)."""

from __future__ import annotations

import base64

from starlette.concurrency import run_in_threadpool

from mminf.api_server.openai._util import now, rid


async def create_images(api, model_name, adapter, req):  # noqa: ARG001
    args = adapter.image_to_request(req, api.upload_dir)
    request_id = rid("img")

    api.submit_request(
        text=args.text,
        file_paths=args.file_paths,
        input_modalities=args.input_modalities,
        output_modalities=["image"],
        model_kwargs=args.model_kwargs,
        streaming=False,
        request_id=request_id,
    )

    chunks = await run_in_threadpool(api.collect_results, request_id)
    data = [
        {"b64_json": base64.b64encode(c.data).decode("ascii"), "url": None}
        for c in chunks
        if c.modality == "image"
    ]
    return {"created": now(), "data": data}
