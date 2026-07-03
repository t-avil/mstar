from dataclasses import dataclass, field

from mstar.graph.base import GraphEdge
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.profile.format import InputInfo, RxInfo, TxInfo
from mstar.profile.worker import GraphTimings


@dataclass
class ResultChunk:
    """One chunk of generated output for a request."""
    request_id: str
    modality: str  # "text" | "image" | "audio" | "video"
    data: bytes  # raw payload (text encoded as utf-8)
    metadata: dict = field(default_factory=dict)


@dataclass
class ResultTensors:
    request_id: str
    modality: str
    graph_edge: GraphEdge
    loop_indices: NestedLoopIndices
    metadata: dict = field(default_factory=dict)


@dataclass
class SlimResultTokens:
    """MSTAR_SLIM_EMIT steady-state item: token values only.

    After the FIRST full ``ResultTensors`` for a (request_id, name) pair has
    been sent (the "template"), later steps of the same inline emit edge carry
    only this — the api server synthesizes a full ``ResultTensors`` from its
    cached template plus these values. Pickling a full GraphEdge per rid per
    step was the bulk of the worker's send_outputs cost (~3.4 ms/step main
    thread at i2t B32).

    MSTAR_SLIM_EMIT2: ``loop_key`` replaces the per-step pickled
    ``NestedLoopIndices`` with plain ints ``(wg_fwd_pass_idx, *values)`` —
    valid ONLY when the sender verified the step's loop layout
    (loop_name_order content + loop_indices key order) still matches the
    template's, so the consumer reconstructs an equal object from the cached
    template. Exactly one of ``loop_indices`` / ``loop_key`` is set.
    """
    request_id: str
    name: str
    values: list
    loop_indices: NestedLoopIndices | None
    loop_key: tuple | None = None


@dataclass
class ResultTensorsBatch:
    """Coalesced inline emit_to_client results for one decode step.

    Carries the qualifying inline-emit ``ResultTensors`` of a single step
    across all requests in the batch, sent as ONE APIServerMessage instead
    of one per request (see MSTAR_BATCH_EMIT). Every item MUST be an
    inline-values result (no transported SHM tensors), so the api-server
    discard path stays a no-op per item. Items can have different
    request_ids and different rid-status on the api side, so each is routed
    individually — this is purely a transport-level fan-in / fan-out.

    With MSTAR_SLIM_EMIT, items may also be ``SlimResultTokens`` — the
    consumer synthesizes the full item from its per-(rid, name) template
    (guaranteed to precede slim items: same FIFO ZMQ stream).
    """
    items: list = field(default_factory=list)


@dataclass
class RequestComplete:
    """Signals that a request has finished processing."""
    request_id: str
    # Maps output signal name to its final forward pass number.
    # The API server waits until all entries are received before
    # completing the request.
    final_outputs: dict[str, NestedLoopIndices]
    conductor_ingest_time: float
    conductor_finish_time: float
    graph_timings: GraphTimings = field(default_factory=dict)
    rx_info: list[RxInfo] = field(default_factory=list)
    tx_info: list[TxInfo] = field(default_factory=list)


@dataclass
class APIServerMessage:
    """Envelope for messages received by the API server."""
    message_type: str  # "result_tensors" | "result_tensors_batch" | "request_complete" | "setup_done"
    body: ResultTensors | ResultTensorsBatch | RequestComplete | None = None  # None for setup_done message


@dataclass
class DataWorkerProfile:
    """Profiling reported by the API-server data worker at preprocess finish:
    the timestamp at which the request was handed to the conductor and the
    per-modality sizes of the raw inputs. (The data worker's tx/rx are read
    directly from its tensor manager at request completion, not via this.)"""
    request_id: str
    preprocess_finish_time: float | None = None  # time.perf_counter
    inputs: list[InputInfo] = field(default_factory=list)


@dataclass
class PreprocessInput:
    request_id: str
    text: str | None

    # file_paths is modality: list of filenames
    file_paths: dict[str, list[str]] | None
    input_modalities: list[str]
    output_modalities: list[str]
    model_kwargs: dict
