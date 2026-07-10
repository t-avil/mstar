

import collections
import logging
import os
import queue
import threading
import time

import torch

from mstar.graph.loop_indices import NestedLoopIndices

try:
    import torchaudio  # noqa: F401 — probes availability; real usage in callers
    from torchcodec.decoders import VideoDecoder
except (ImportError, RuntimeError, OSError):
    VideoDecoder = None

from mstar.api_server.request_types import (
    DataWorkerProfile,
    PreprocessInput,
    ResultChunk,
    ResultTensors,
)
from mstar.communication.communicator import CommProtocol, ZMQCommunicator
from mstar.communication.tensors import NameToTensorList, create_tensor_communication_manager
from mstar.model.base import Model
from mstar.profile.format import InputInfo, RxInfo, TxInfo
from mstar.utils.ipc_format import (
    AbortRequest,
    ConductorMessage,
    ConductorMessageType,
    NewRequestConductor,
    TensorReceived,
    UnpersistTensors,
    WorkerMessageType,
)

logger = logging.getLogger(__name__)


def _preprocess_loop(**kwargs):
    worker = PreprocessWorkerThread(**kwargs)
    worker.run()


NameToLoopIndices = dict[str, NestedLoopIndices]


class PreprocessWorker:
    def __init__(
        self,
        model: Model | None = None,
        hostname: str = "localhost",
        socket_path_prefix: str = "/tmp/mstar",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
        tcp_transfer_device="",
        enable_prof: bool=False
    ):
        self.request_input_queue = queue.Queue()
        self.result_tensor_input_queue = queue.Queue()
        self.cleanup_request_queue = queue.Queue()
        self.abort_request_queue = queue.Queue()
        self.discard_tensor_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.profile_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.per_request_reading_tensors = {}
        self.output_loop_idxs: dict[str, NameToLoopIndices] = {}

        # Build the communicator + tensor manager here (main thread) and hand
        # them to the worker thread, rather than constructing them inside it.
        # The socket is only *used* from the worker thread, but owning the
        # tensor manager here lets the main thread read its tx/rx profiling
        # directly once a request is done (no cross-thread queue / race).
        self.communicator = ZMQCommunicator(
            my_id="api_server_preprocess_worker",
            push_ids=["conductor"],
            ipc_socket_path_prefix=socket_path_prefix,
        )  # only used to send (from the worker thread)
        self.tensor_manager = create_tensor_communication_manager(
            protocol=tensor_comm_protocol,
            my_entity_id="api_server_preprocess_worker",
            hostname=hostname,
            device="cpu",
            communicator=self.communicator,
            tcp_transfer_device=tcp_transfer_device,
            enable_prof=enable_prof,
        )

        self.thread = threading.Thread(
            target=_preprocess_loop,
            kwargs=dict(
                in_queue=self.request_input_queue,
                result_tensor_queue=self.result_tensor_input_queue,
                out_queue=self.output_queue,
                profile_queue=self.profile_queue,
                cleanup_request_queue=self.cleanup_request_queue,
                abort_request_queue=self.abort_request_queue,
                discard_tensor_queue=self.discard_tensor_queue,
                stop_event=self.stop_event,
                communicator=self.communicator,
                tensor_manager=self.tensor_manager,
                model=model,
                enable_prof=enable_prof
            )
        )
        self.thread.start()

    def new_request(self, input: PreprocessInput):
        self.output_loop_idxs[input.request_id] = {}
        self.per_request_reading_tensors[input.request_id] = 0
        self.request_input_queue.put(input)

    def abort_request(self, request_id: str):
        self.abort_request_queue.put(request_id)
        self.cleanup_request(request_id)

    def new_result_tensors(self, input: ResultTensors):
        name = input.graph_edge.name
        if input.request_id not in self.output_loop_idxs:
            # Request was removed while this output was still in flight; ack the
            # tensors so the producing worker can reclaim them rather than leak.
            logger.debug("Late result_tensors for cleaned-up request %s, acking and dropping", input.request_id)
            self.discard_result_tensors(input)
            return

        self.output_loop_idxs[input.request_id][name] = input.loop_indices.max(
            self.output_loop_idxs[input.request_id].get(name, None)
        )

        self.per_request_reading_tensors[input.request_id] += len(input.graph_edge.tensor_info)
        logger.debug(
            "Data worker reading queue for request %s increased to length %d",
            input.request_id,  self.per_request_reading_tensors[input.request_id]
        )
        self.result_tensor_input_queue.put(input)

    def discard_result_tensors(self, input: ResultTensors):
        """Ack and drop result tensors for an already-removed request.

        Routed to the worker thread (which owns the communicator) so the
        producing worker gets its TENSOR_RECEIVED ack and frees the buffers.
        """
        self.discard_tensor_queue.put(input)

    def has_pending_tensors(self, request_id: str):
        return self.per_request_reading_tensors.get(request_id, 0) > 0

    def received_final_chunks(
        self, request_id: str,
        final_outputs: dict[str, NestedLoopIndices],
    ):
        return all(
            not loop_iters.label_context_gt( # recv'd loop iters is not less than the final_fwd
                self.output_loop_idxs[request_id].get(name, None)
            ) for name, loop_iters in final_outputs.items()
        )

    def get_result_chunks(self)-> list[ResultChunk]:
        results = []
        while not self.output_queue.empty():
            result: ResultChunk = self.output_queue.get()
            self.per_request_reading_tensors[result.request_id] -= 1
            logger.debug(
                "Data worker reading queue for request %s decreased to length %d",
                result.request_id,  self.per_request_reading_tensors[result.request_id]
            )
            results.append(result)
        return results

    def get_profile_updates(self) -> list[DataWorkerProfile]:
        """Drain preprocess-side profiling updates emitted by the worker thread."""
        updates = []
        while not self.profile_queue.empty():
            updates.append(self.profile_queue.get())
        return updates

    def get_tx_info(self, request_id: str) -> list[TxInfo]:
        """Snapshot the data worker's send (tx) profiling for a request.

        Safe to call from the main thread once the request is done: by then the
        worker thread is no longer mutating this request's tx state, and the
        caller must read it before ``cleanup_request`` drops it.
        """
        return self.tensor_manager.get_tx_info(request_id)

    def get_rx_info(self, request_id: str) -> list[RxInfo]:
        """Snapshot the data worker's receive (rx) profiling for a request.

        Same safety contract as :meth:`get_tx_info` — read once the request's
        final chunks have all arrived, before ``cleanup_request``.
        """
        return self.tensor_manager.get_rx_info(request_id)

    def cleanup_request(self, request_id: str):
        self.cleanup_request_queue.put(request_id)
        self.output_loop_idxs.pop(request_id, None)
        self.per_request_reading_tensors.pop(request_id, None)

    def shutdown(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join()


class PreprocessWorkerThread:
    def __init__(
        self,
        in_queue: queue.Queue, # for preprocessing
        result_tensor_queue: queue.Queue, # for output streaming
        out_queue: queue.Queue,
        profile_queue: queue.Queue,
        cleanup_request_queue: queue.Queue,
        abort_request_queue: queue.Queue,
        discard_tensor_queue: queue.Queue,
        stop_event: threading.Event,
        communicator: ZMQCommunicator,
        tensor_manager,
        device: str = "cpu",
        model: Model | None = None,
        enable_prof: bool=False
    ):
        self.in_queue = in_queue
        self.result_tensor_queue = result_tensor_queue
        self.cleanup_request_queue = cleanup_request_queue
        self.abort_request_queue = abort_request_queue
        self.discard_tensor_queue = discard_tensor_queue
        self.out_queue = out_queue
        self.profile_queue = profile_queue

        self.stop_event = stop_event
        self.device = device
        self.model = model
        self.enable_prof = enable_prof

        self.tensor_uuid_to_metadata_per_request = {}

        # Owned by PreprocessWorker (main thread); used only from this thread.
        self.communicator = communicator
        self.tensor_manager = tensor_manager

        # MSTAR_ORDERED_EMIT: emit ResultChunks per (rid, modality) in ARRIVAL
        # order rather than read-completion order. Without this, a mixed
        # inline/SHM stream reorders: inline items (decode new-token ints)
        # emit synchronously in _read_result_tensor while an SHM item (the
        # prefill step's first token — its uuid also feeds the prefill→decode
        # loop-back edge, so it is excluded from the inline transport and must
        # be fetched) emits only when its async read completes. Under load the
        # fetch lands 1..k decode tokens late, so the client stream shows the
        # FIRST generated token displaced mid-sentence (or, when the request
        # finishes first, missing entirely). Default OFF = current behavior.
        self._ordered_emit = os.environ.get("MSTAR_ORDERED_EMIT", "0") == "1"
        self._ordered_emit_debug = (
            os.environ.get("MSTAR_ORDERED_EMIT_DEBUG", "0") == "1"
        )
        # (rid, modality) -> deque of entries in arrival order. Entry:
        # {"ready": bool, "uuid_order": [uuid,...], "chunks": {uuid: chunk},
        #  "pending": set[uuid]}   (inline entries: ready=True, uuid_order
        # ordered as built, pending empty).
        self._emit_fifos: dict[tuple[str, str], collections.deque] = {}
        # uuid -> (fifo_key, entry) so read completions find their entry.
        self._uuid_to_emit_entry: dict[tuple[str, str], tuple] = {}

    def _process_input(
        self, input: PreprocessInput
    ):
        tensors: NameToTensorList = {}
        input_metadata = {}

        # First, load raw modality tensors from file_paths (images, audio, video)
        # so they can be passed to process_prompt() below.
        if input.file_paths is not None:
            for modality in input.file_paths:
                key = f"{modality}_inputs"
                tensors[key] = []
                # TODO: maybe make a class of tensors_and_metadata later (figure out how to use metadata)
                input_metadata[key] = []

                for filepath in input.file_paths[modality]:
                    # ---- Image ----
                    if modality == "image":
                        out = self.model.load_image(filepath, self.device)
                        tensors[key].append(out.data)
                        input_metadata[key].append(out.metadata)

                    # ---- Audio ----
                    elif modality == "audio":
                        out = self.model.load_audio(filepath, self.device)
                        tensors[key].append(out.data)
                        input_metadata[key].append(out.metadata)

                    # ---- Video ----
                    elif modality == "video":
                        out = self.model.load_video(filepath, self.device)
                        tensors[key].append(out.data)
                        input_metadata[key].append(out.metadata)


        # Then, tokenize the prompt and let the model augment/transform the
        # tensors dict (e.g., Qwen3-Omni needs to compute pixel_values,
        # image_grid_thw, audio_features, audio_seqlens from the raw tensors
        # loaded above).  process_prompt receives the raw multimodal tensors
        # and returns any additional tensors to merge into the final dict.
        if self.model is not None:
            prompt_tensors = self.model.process_prompt(
                input.text,
                input.input_modalities,
                input.output_modalities,
                tensors=tensors,
                input_metadata=input_metadata,
                **(input.model_kwargs or {}),
            )
            if prompt_tensors:
                tensors.update(prompt_tensors)
        elif input.text is not None:
            # Fallback: encode as UTF-8 bytes -> uint8 tensor
            byte_data = input.text.encode("utf-8")
            tensors["text_inputs"] = [torch.tensor(
                list(byte_data), dtype=torch.uint8, device=self.device
            )]

        initial_signals = self.tensor_manager.store_and_return_tensor_info(
            request_id=input.request_id,
            tensors=tensors # dict(modality_input: list[tensors])
        )
        all_uuids = sum([
            [info.uuid for info in infos] for infos in initial_signals.values()
        ], start=[])
        self.tensor_manager.register_for_send(
            request_id=input.request_id,
            uuids=all_uuids
        )
        # also persist all of the input signals
        for uuid in all_uuids:
            self.tensor_manager.set_persist(
                input.request_id, uuid, persist=True
            )

        msg = ConductorMessage(
            message_type=ConductorMessageType.NEW_REQUEST,
            body=NewRequestConductor(
                request_id=input.request_id,
                initial_signals=initial_signals,
                initial_input_modalities=input.input_modalities,
                initial_output_modalities=input.output_modalities,
                input_metadata=input_metadata,
                model_kwargs=input.model_kwargs
            ),
        )
        self.communicator.send("conductor", msg)

        # Record preprocess-side profiling: the moment the fully preprocessed
        # request was handed off to the conductor, plus the per-modality sizes of
        # the *raw* inputs. ``perf_counter`` is consistent here because the worker
        # runs as a thread inside the API server process. (tx/rx are snapshotted
        # directly by the main thread at request completion — see APIServer.)
        if self.enable_prof:
            self.profile_queue.put(DataWorkerProfile(
                request_id=input.request_id,
                preprocess_finish_time=time.perf_counter(),
                inputs=self._summarize_inputs(input),
            ))

    @staticmethod
    def _summarize_inputs(input: PreprocessInput) -> list[InputInfo]:
        """Aggregate the *raw* (pre-decoding) inputs into per-modality sizes.

        Reports the bytes the client actually sent — uploaded file sizes on
        disk and the UTF-8 length of the prompt — rather than the much larger
        decoded tensors (e.g. a compressed JPEG vs. its raw RGB tensor), so the
        numbers line up with what a user thinks of as "input size".
        """
        infos = []
        if input.text:
            infos.append(InputInfo(
                modality="text",
                count=1,
                total_bytes=len(input.text.encode("utf-8")),
            ))
        for modality, paths in (input.file_paths or {}).items():
            total_bytes = 0
            for path in paths:
                try:
                    total_bytes += os.path.getsize(path)
                except OSError:
                    pass  # file already cleaned up / unreadable — count as 0
            infos.append(InputInfo(
                modality=modality,
                count=len(paths),
                total_bytes=total_bytes,
            ))
        return infos

    def _read_result_tensor(
        self, result: ResultTensors
    ):
        result.graph_edge.name = f"{result.modality}_output"
        # Inline fast path: token values arrived in the message metadata, so
        # there is no SHM tensor to fetch and no producer ack to send. The
        # producer already released its tensor_store ref locally.
        if result.metadata and "inline_values" in result.metadata:
            if self._ordered_emit:
                # Enqueue at the FIFO tail; emits only once every earlier
                # arrival for this (rid, modality) has emitted.
                key = (result.request_id, result.modality)
                chunks = self._build_inline_chunks(result)
                entry = {
                    "ready": True,
                    "uuid_order": list(range(len(chunks))),
                    "chunks": dict(enumerate(chunks)),
                    "pending": set(),
                }
                self._emit_fifos.setdefault(key, collections.deque()).append(entry)
                if self._ordered_emit_debug:
                    logger.warning(
                        "ORDEMIT arrival INLINE rid=%s n_chunks=%d fifo_len=%d",
                        result.request_id, len(chunks),
                        len(self._emit_fifos[key]),
                    )
                self._flush_emit_fifo(key)
            else:
                self._emit_inline_result(result)
            return
        self.tensor_manager.start_read_tensors(
            request_id=result.request_id,
            graph_edges=[result.graph_edge],
        )
        if result.request_id not in self.tensor_uuid_to_metadata_per_request:
            self.tensor_uuid_to_metadata_per_request[result.request_id] = {}
        for tensor_info in result.graph_edge.tensor_info:
            self.tensor_uuid_to_metadata_per_request[result.request_id][
                tensor_info.uuid] = result.metadata
        if self._ordered_emit:
            key = (result.request_id, result.modality)
            uuids = [info.uuid for info in result.graph_edge.tensor_info]
            entry = {
                # A signal-only emit edge (no tensor_info — e.g. the leading
                # text_output marker) transports nothing: it is trivially
                # ready, else it wedges the FIFO head forever (no read will
                # ever complete it) and every later chunk is held until the
                # request's TTL drops them (observed: all-empty responses).
                "ready": not uuids,
                "uuid_order": uuids,
                "chunks": {},
                "pending": set(uuids),
            }
            self._emit_fifos.setdefault(key, collections.deque()).append(entry)
            for u in uuids:
                waiters = self._uuid_to_emit_entry.setdefault(
                    (result.request_id, u), []
                )
                if waiters and self._ordered_emit_debug:
                    logger.warning(
                        "ORDEMIT alias (multi-waiter) rid=%s uuid=%s n=%d",
                        result.request_id, u, len(waiters) + 1,
                    )
                # List-valued: the SAME uuid can be referenced by MULTIPLE
                # arrival entries (aliased emit edges / re-sends); one read
                # completion must satisfy every waiter or the orphaned
                # earlier entry wedges the FIFO head forever.
                waiters.append((key, entry))
            if self._ordered_emit_debug:
                logger.warning(
                    "ORDEMIT arrival SHM rid=%s uuids=%s fifo_len=%d",
                    result.request_id, uuids, len(self._emit_fifos[key]),
                )
            if entry["ready"]:
                # Signal-only entry: pop it (and any ready run) promptly so
                # it never lingers at the head.
                self._flush_emit_fifo(key)

    def _emit_inline_result(self, result: ResultTensors):
        """Produce ResultChunk(s) directly from inline token values.

        Mirrors _process_read_tensors' emission but skips the transport
        fetch: one chunk per tensor_info entry (so per_request_reading_tensors,
        bumped by len(tensor_info) in new_result_tensors, balances exactly),
        each reconstructed as a byte-identical tensor from the inline ints
        using the tensor_info dtype/shape and run through the same postprocess.
        """
        for chunk in self._build_inline_chunks(result):
            self.out_queue.put(chunk)

    def _build_inline_chunks(self, result: ResultTensors) -> list[ResultChunk]:
        """Construct the ResultChunk list for an inline-values message
        (shared by the immediate path and MSTAR_ORDERED_EMIT's FIFO path)."""
        modality = result.graph_edge.name.replace("_output", "")
        # The producer keys inline_values by the pre-rename edge name; there is
        # exactly one entry (this edge). Fall back to the single value list.
        inline_map: dict = result.metadata["inline_values"]
        values = next(iter(inline_map.values())) if inline_map else []
        chunk_metadata = {
            k: v for k, v in (result.metadata or {}).items()
            if k != "inline_values"
        }
        chunks: list[ResultChunk] = []
        for tensor_info in result.graph_edge.tensor_info:
            n = 1
            for d in tensor_info.dims:
                n *= int(d)
            ints = values[:n]
            values = values[n:]
            tensor = torch.tensor(ints, dtype=tensor_info.dtype).reshape(
                tensor_info.dims
            )
            postprocessed = self.model.postprocess(tensor, modality)
            chunks.append(ResultChunk(
                request_id=result.request_id,
                modality=modality,
                data=postprocessed,
                metadata=chunk_metadata,
            ))
        return chunks

    def _flush_emit_fifo(self, key: tuple[str, str]) -> None:
        """Emit the head-run of ready entries for one (rid, modality) FIFO.

        Arrival order == the producing worker's send order (single ZMQ FIFO
        per rid), so draining ready heads preserves true token order; an
        unread SHM entry at the head holds everything behind it until its
        read lands (at most the transport latency — the same latency that
        today reorders instead).
        """
        fifo = self._emit_fifos.get(key)
        if fifo is None:
            return
        n_emitted = 0
        while fifo and fifo[0]["ready"]:
            entry = fifo.popleft()
            for u in entry["uuid_order"]:
                chunk = entry["chunks"].get(u)
                if chunk is not None:
                    self.out_queue.put(chunk)
                    n_emitted += 1
        if self._ordered_emit_debug:
            logger.warning(
                "ORDEMIT flush key=%s emitted=%d held=%d head_pending=%s",
                key, n_emitted, len(fifo),
                (sorted(fifo[0]["pending"]) if fifo else None),
            )
        if not fifo:
            self._emit_fifos.pop(key, None)

    def _discard_result_tensor(
        self, result: ResultTensors
    ):
        # Inline messages carry no transported tensors: the producer never
        # registered them for send and already released its ref locally, so
        # there is nothing to ack. Acking would deref uuids the producer
        # doesn't hold — make discard a no-op for inline-only messages.
        if result.metadata and "inline_values" in result.metadata:
            return
        # The request is gone, so don't start a read — just ack the tensors back
        # to the producing worker so it can free the source buffers.
        self.tensor_manager.ack_unread_tensors(
            request_id=result.request_id,
            graph_edges=[result.graph_edge],
        )

    def _process_read_tensors(self):
        did_work = False
        for request_id, graph_edges in self.tensor_manager.get_ready_tensors().items():
            did_work = True
            for graph_edge in graph_edges:
                modality = graph_edge.name.replace("_output", "")

                for tensor_info in graph_edge.tensor_info:
                    logger.debug("Reading in OUTPUT tensor %s with uuid %s", graph_edge.name, tensor_info.uuid)
                    tensor = self.tensor_manager.get_tensor(
                        request_id=request_id,
                        uuid=tensor_info.uuid
                    )
                    postprocessed = self.model.postprocess(
                        tensor, modality
                    )

                    chunk_metadata = self.tensor_uuid_to_metadata_per_request[request_id][
                        tensor_info.uuid] or {}
                    # Audio is emitted as headerless 16-bit PCM; surface the
                    # model's output sample rate so clients can wrap it.
                    if modality == "audio" and self.model is not None:
                        chunk_metadata = {
                            **chunk_metadata,
                            "sample_rate": self.model.get_output_sample_rate("audio"),
                        }

                    chunk = ResultChunk(
                        request_id=request_id,
                        modality=modality,
                        data=postprocessed,
                        metadata=chunk_metadata,
                    )
                    waiters = (
                        self._uuid_to_emit_entry.pop(
                            (request_id, tensor_info.uuid), None,
                        )
                        if self._ordered_emit else None
                    )
                    if self._ordered_emit and self._ordered_emit_debug:
                        logger.warning(
                            "ORDEMIT completion rid=%s uuid=%s waiters=%s",
                            request_id, tensor_info.uuid,
                            len(waiters) if waiters else 0,
                        )
                    if waiters:
                        # MSTAR_ORDERED_EMIT: attach to every arrival-ordered
                        # entry waiting on this uuid; flush each FIFO whose
                        # entry became ready at the head.
                        for key, entry in waiters:
                            entry["chunks"][tensor_info.uuid] = chunk
                            entry["pending"].discard(tensor_info.uuid)
                            if not entry["pending"]:
                                entry["ready"] = True
                                self._flush_emit_fifo(key)
                    else:
                        # Flag off — or an entry dropped by request cleanup
                        # (chunk is late for a dead rid; emit as before, the
                        # main thread's get_result_chunks tolerates it).
                        self.out_queue.put(chunk)
                    del self.tensor_uuid_to_metadata_per_request[request_id][
                        tensor_info.uuid]
                    self.tensor_manager.dereference(
                        request_id=request_id,
                        uuid=tensor_info.uuid
                    )
        return did_work

    def _process_messages(self):
        did_work = False
        for message in self.communicator.get_all_new_messages():
            did_work = True
            if message.message_type == WorkerMessageType.TENSOR_RECEIVED:
                body: TensorReceived = message.body
                for (uuid, ref_cnt) in body.successful_tensors.items():
                    self.tensor_manager.dereference(
                        body.request_id, uuid, n=ref_cnt
                    )
            elif message.message_type == WorkerMessageType.UNPERSIST_TENSORS:
                body: UnpersistTensors = message.body
                for (uuid, ref_cnt) in body.uuid_to_ref_count.items():
                    self.tensor_manager.increment_ref(
                        body.request_id, uuid, n=ref_cnt
                    )
                    self.tensor_manager.set_persist(
                        body.request_id, uuid, persist=False
                    )
        return did_work

    def run(self):
        while not self.stop_event.is_set():
            did_work = False
            try:
                did_work = self._process_messages()
                if not self.in_queue.empty():
                    did_work = True
                    self._process_input(self.in_queue.get())
                if not self.result_tensor_queue.empty():
                    did_work = True
                    self._read_result_tensor(self.result_tensor_queue.get())
                if not self.abort_request_queue.empty():
                    did_work = True
                    self.communicator.send(
                        "conductor",
                        ConductorMessage(
                            message_type=ConductorMessageType.ABORT_REQUEST,
                            body=AbortRequest(request_id=self.abort_request_queue.get()),
                        ),
                    )
                if not self.discard_tensor_queue.empty():
                    did_work = True
                    self._discard_result_tensor(self.discard_tensor_queue.get())
                if not self.cleanup_request_queue.empty():
                    did_work = True
                    req_id = self.cleanup_request_queue.get()
                    self.tensor_manager.cleanup_request(req_id)
                    if req_id in self.tensor_uuid_to_metadata_per_request:
                        del self.tensor_uuid_to_metadata_per_request[req_id]
                    if self._ordered_emit:
                        # Drop this rid's held-back entries and uuid refs; a
                        # late read for a dropped entry falls back to the
                        # direct out_queue path above (harmless for dead rid).
                        for key in [
                            k for k in self._emit_fifos if k[0] == req_id
                        ]:
                            if self._ordered_emit_debug:
                                fifo = self._emit_fifos.get(key)
                                logger.warning(
                                    "ORDEMIT cleanup-drop key=%s held=%d "
                                    "head_pending=%s",
                                    key, len(fifo) if fifo else 0,
                                    (sorted(fifo[0]["pending"])
                                     if fifo else None),
                                )
                            self._emit_fifos.pop(key, None)
                        for uk in [
                            k for k in self._uuid_to_emit_entry
                            if k[0] == req_id
                        ]:
                            self._uuid_to_emit_entry.pop(uk, None)
                did_work = did_work or self._process_read_tensors()
            except Exception:
                logger.exception("PreprocessWorkerThread error")

            if not did_work:
                time.sleep(0.001)

