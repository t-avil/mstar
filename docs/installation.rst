Installation
============

Requirements
------------

- **Python 3.12+**.
- **Linux with an NVIDIA GPU** and a recent **CUDA** toolkit for the GPU model
  families. A CPU-only machine can exercise the graph/worker plumbing in *dummy mode*
  (submodules return ``None``) for development and the modular tests, but not real model
  inference.
- Enough GPU memory for the model you intend to serve — several families (e.g.
  BAGEL-7B, Qwen3-Omni-30B) are multi-GPU-class models.

Install from source
-------------------

``mminf`` is installed from source in editable mode:

.. code-block:: bash

   git clone https://github.com/merceod/multimodal_inference.git
   cd multimodal_inference
   pip install -e .

This pulls in the core runtime (PyTorch, FastAPI/Uvicorn, ZMQ, …) and installs the two
console scripts, ``mminf`` and ``mminf-serve``.

Optional dependencies
---------------------

Model families and some output formats need extra packages, exposed as pip *extras*:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Extra
     - Installs / use for
   * - ``.[bagel]``
     - BAGEL runtime: ``transformers``, ``flashinfer-python``, ``safetensors``,
       ``einops``, ``Pillow``, ``torchvision`` / ``torchaudio`` / ``torchcodec``,
       ``huggingface-hub``, ``regex``, and ``mooncake-transfer-engine`` (RDMA transport).
   * - ``.[qwen3_omni]``
     - Qwen3-Omni runtime: the BAGEL set plus ``flash-attn``, ``qwen-omni-utils``,
       ``sgl-kernel``, and ``datasets``.
   * - ``.[orpheus]``
     - Orpheus TTS runtime: ``transformers``, ``flashinfer-python``, ``safetensors``,
       ``einops``, ``huggingface-hub``, ``mooncake-transfer-engine``.
   * - ``.[pi05]``
     - Pi0.5 runtime: ``transformers``, ``flashinfer-python``, ``safetensors``,
       ``triton``, ``huggingface-hub``, ``mooncake-transfer-engine``.
   * - ``.[vjepa2]`` / ``.[vjepa2_ac]``
     - V-JEPA 2 runtime: ``safetensors``, ``torchcodec``, ``huggingface-hub``,
       ``mooncake-transfer-engine`` (``vjepa2_ac`` also adds ``flashinfer-python``).
   * - ``.[audio]``
     - ``soundfile`` — only needed to return **non-WAV** audio containers (mp3/flac/…)
       from the OpenAI/SDK audio surfaces. WAV/PCM output works without it.
   * - ``.[dev]``
     - ``ruff`` + ``pytest`` for linting and the test suite.

Combine extras as needed:

.. code-block:: bash

   pip install -e ".[bagel,audio,dev]"

.. note::

   ``torch``, ``torchvision``, and ``torchaudio`` are already in the base install; each
   model extra adds that family's remaining runtime — FlashInfer for the autoregressive
   backbones, Transformers, safetensors, any codec/media libraries, and the Mooncake RDMA
   transport for disaggregated deployments.

GPU libraries
-------------

The GPU model families depend on:

- **FlashInfer** (``flashinfer-python``) — paged attention and continuous batching for the
  autoregressive backbones (every model with a ``KV_CACHE`` node runs attention through it).
- **flash-attn** — used by Qwen3-Omni.
- **mooncake-transfer-engine** — RDMA tensor transport for multi-GPU, disaggregated
  deployments. Single-node deployments can use shared-memory (``SHM``) or ``TCP`` transport
  instead (see :doc:`serving`).

These are installed by the extras above. Make sure your installed ``torch`` matches your
CUDA version *before* installing them.

Verify the install
------------------

.. code-block:: bash

   python -c "import mminf; print('mminf import OK')"
   mminf --help
   mminf-serve --help

Next: :doc:`quickstart`.
