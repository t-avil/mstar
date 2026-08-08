"""Order-preserving multimodal prompt adapter.

Intake hands models an ordered list of :class:`PromptPart` — the user's text
and attachments in the sequence they were written — instead of a bag of
per-modality lists. From that, models derive a *prefill plan*: the ordered
sequence of text spans and modality items to prefill.

Two ways to reach a plan, both ending in the same type:

* placeholder scan (Qwen3-Omni) — render the prompt once with the processor's
  own per-item placeholders, tokenize once, then read the placement back off
  the token ids with :func:`find_media_spans` / :func:`split_around_spans`.
  One tokenizer call, so no BPE drift at span boundaries.
* per-segment tokenization (BAGEL) — tokenize each text span on its own.

:func:`prefill_plan` is the single source of truth for segment order: both the
prompt-building side and the schedule-building side call it, so they cannot
disagree about how many text spans there are or where they sit.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

TEXT = "text"


@dataclass(frozen=True)
class PromptPart:
    """One element of a prompt, in request order.

    ``index`` is the position within its own modality's input list, so a media
    part addresses ``tensors[f"{modality}_inputs"][index]``.
    """

    modality: str
    text: str | None = None
    index: int = 0


@dataclass(frozen=True)
class MediaSpan:
    """A ``<|x_start|> pad* <|x_end|>`` run located in a tokenized prompt.

    The bounds are sentinel-inclusive because a modality walk emits the
    sentinel embeddings itself alongside the encoder output.
    """

    modality: str
    index: int
    start: int
    stop: int


def parts_from_modalities(
    input_modalities: list[str], texts: Iterable[str] | str | None = None
) -> list[PromptPart]:
    """Rebuild the parts for a layout, filling the text slots from ``texts``.

    ``input_modalities`` is one entry per part, in order, and is the layout
    every consumer plans from — the schedule builder never sees anything else.
    Rebuilding from it rather than carrying a second copy of the ordering is
    what keeps the prompt and the schedule from disagreeing. Text slots left
    unfilled carry ``None``, which is enough to plan with.
    """
    if isinstance(texts, str):
        texts = [texts]
    remaining = iter(texts or ())
    parts: list[PromptPart] = []
    seen: dict[str, int] = {}
    for modality in input_modalities:
        if modality == TEXT:
            parts.append(PromptPart(modality=TEXT, text=next(remaining, None)))
        else:
            index = seen.get(modality, 0)
            parts.append(PromptPart(modality=modality, index=index))
            seen[modality] = index + 1
    return parts


def prefill_plan(
    parts: list[PromptPart], *, leading_text: bool = True
) -> list[PromptPart]:
    """Order the prefill segments for ``parts``.

    Adjacent text collapses into one segment, because the rendered prompt puts
    it in one contiguous run. A trailing text segment always exists — the
    template's turn close — even when the user supplied no text on that side.
    ``leading_text`` says whether the template also opens with one before the
    first attachment; BAGEL's generation prompt does not. Text segments are
    numbered in order, so ``index`` addresses the model's own list of text
    spans.
    """
    plan: list[PromptPart] = []
    n_text = 0
    pending = leading_text
    for part in parts:
        if part.modality == TEXT:
            pending = True
            continue
        if pending:
            plan.append(PromptPart(modality=TEXT, index=n_text))
            n_text += 1
            pending = False
        plan.append(part)
    plan.append(PromptPart(modality=TEXT, index=n_text))
    return plan


def find_media_spans(
    input_ids: torch.Tensor, specs: dict[str, tuple[int, int, int]]
) -> list[MediaSpan]:
    """Locate every placeholder run in ``input_ids``, in token order.

    ``specs`` maps a modality to its ``(start_id, pad_id, end_id)`` sentinel
    triple. Runs are returned with per-modality running indices, so the nth
    image span addresses the nth image input.
    """
    by_pad = {pad: (modality, start, end) for modality, (start, pad, end) in specs.items()}
    ids = input_ids.tolist()
    spans: list[MediaSpan] = []
    seen: dict[str, int] = {}
    i, n = 0, len(ids)
    while i < n:
        entry = by_pad.get(ids[i])
        if entry is None:
            i += 1
            continue
        modality, start_id, end_id = entry
        j = i
        while j < n and ids[j] == ids[i]:
            j += 1
        if i == 0 or ids[i - 1] != start_id or j >= n or ids[j] != end_id:
            raise ValueError(
                f"{modality} placeholder run at {i}..{j} is not wrapped in its "
                "start/end sentinels"
            )
        index = seen.get(modality, 0)
        spans.append(MediaSpan(modality=modality, index=index, start=i - 1, stop=j + 1))
        seen[modality] = index + 1
        i = j + 1
    return spans


def split_around_spans(
    input_ids: torch.Tensor, spans: list[MediaSpan]
) -> list[torch.Tensor]:
    """Return the text between the spans, in order, dropping empty pieces.

    Two adjacent attachments leave nothing between them; a zero-length prefill
    walk has nothing to embed, so that piece is not emitted. :func:`prefill_plan`
    drops the same pieces, by the same rule.
    """
    segments: list[torch.Tensor] = []
    cursor = 0
    for span in spans:
        if span.start > cursor:
            segments.append(input_ids[cursor:span.start])
        cursor = span.stop
    if cursor < len(input_ids):
        segments.append(input_ids[cursor:])
    return segments


def check_plan(plan: list[PromptPart], spans: list[MediaSpan], n_text: int) -> None:
    """Fail loudly when the rendered prompt disagrees with the plan.

    A mismatch means an attachment was dropped or reordered between intake and
    tokenization — the failure this adapter exists to make impossible.
    """
    planned = [(p.modality, p.index) for p in plan if p.modality != TEXT]
    scanned = [(s.modality, s.index) for s in spans]
    if planned != scanned:
        raise ValueError(
            f"multimodal placement mismatch: planned {planned}, prompt has {scanned}"
        )
    planned_text = sum(1 for p in plan if p.modality == TEXT)
    if planned_text != n_text:
        raise ValueError(
            f"multimodal placement mismatch: planned {planned_text} text spans, "
            f"prompt has {n_text}"
        )
