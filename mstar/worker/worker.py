import logging
import os
import sys
import threading
import time
import time as _time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from time import sleep

import torch

from mstar.api_server.request_types import APIServerMessage, ResultTensors, ResultTensorsBatch
from mstar.communication.communicator import CommProtocol, ZMQCommunicator
from mstar.communication.event import EventWakeup
from mstar.communication.tensors import NameToTensorList, create_tensor_communication_manager
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig
from mstar.distributed.communication import WorkerTPGroups
from mstar.engine.base import EngineType, NodeBatch, NodeOutput
from mstar.engine.kv_store import KVCacheConfig, StoreWritePolicy, TransferEngineInfo
from mstar.graph.base import GraphEdge, GraphNode
from mstar.graph.graph_io import format_graph_edge_list
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.model.base import Model, WorkerGraph
from mstar.profile.worker import WorkerProfileInfo
from mstar.streaming.stream_buffer import StreamBuffer
from mstar.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    InputSignals,
    MessageSource,
    NewRequest,
    RemoveRequest,
    ScheduleTPNode,
    SetupDone,
    StopLoops,
    TensorReceived,
    UnpersistTensors,
    WorkerGraphsDone,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.utils.profiler import range_pop, range_push
from mstar.worker.engine_manager import EngineManager
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mstar.worker.node_manager_utils import (
    NodeOutputRouting,
    WorkerGraphQueues,
    WorkerGraphsManager,
)

logger = logging.getLogger(__name__)


@dataclass
class PendingBatch:
    batch: ScheduledBatch
    node_batch: NodeBatch
    node_name: str
    partition: str
    graph_walk: str
    future: Future
    speculative_new_iter: bool = False
    loop_name: str = None

@dataclass
class PendingSide:
    """A prefill/encoder batch executing on the side stream + side executor,
    concurrent with the decode chain. Distinct from PendingBatch: it never
    speculates, never loops back, and its postprocess runs opportunistically
    on the main thread. See Worker.run() (MSTAR_SIDE_PREFILL)."""
    batch: ScheduledBatch
    node_batch: NodeBatch
    node_name: str
    partition: str
    graph_walk: str
    future: Future

@dataclass
class Speculation:
    scheduled_batch: ScheduledBatch
    node_batch: NodeBatch
    # ``(name, next_node)`` pairs the spec batch consumed from batch_N's
    # outputs. Two cases:
    #   * Same-node loop-back (AR decode iter K → iter K+1): pairs are
    #     ``{(name, batch_N.node_name) for name in loop_back_outputs}``.
    #   * Forward node A -> node B transition: pairs are
    #     ``{(edge.name, edge.next_node) for edge in batch_N.outputs if
    #     edge.next_node == spec_target.node_name}``.
    # Consumed in ``_thread_outputs_to_speculative`` to splice batch_N's
    # outputs into the spec batch's per-rid input tensors.
    consumed_edges: set[tuple[str, str]]
    continuing_rids: set[str]
    partition: str
    is_new_iter: bool
    is_same_node: bool
    # rid -> edges
    consumed_streaming_edges: dict[str, list[GraphEdge]] = field(default_factory=dict)
    is_yield_away: bool = False
    loop_name: str | None = None
    dropped: set[str] = field(default_factory=set)

    plan_future: Future | None = None


@dataclass(frozen=True)
class PendingLoopStop:
    rid: str
    graph_walk: str
    loop_name: str


class EvictionPolicy(Enum):
    """Strategy for choosing which request to offload to CPU on OOM."""
    LRU = "lru"              # least-recently-used (by execution time)
    MOST_PAGES = "most_pages"  # request holding the most GPU pages


class Worker:
    """
    Real worker that integrates WorkerGraphsManager, EngineManager,
    MicroScheduler, and MooncakeCommunicationManager to execute
    computation via engines.
    """

    def __init__(
        self,
        worker_id: str,
        worker_ids: list[str],
        model: Model,
        my_worker_graphs: list[WorkerGraph],
        kv_config: dict[str, KVCacheConfig],
        model_config: dict,
        all_worker_graph_ids_to_graph_walks: dict[str, set[str]],
        all_worker_graph_ids_to_nodes: dict[str, set[str]],
        all_worker_graph_ids_to_dyn_loops: dict[str, set[str]],
        sharding_config: ShardingConfig,
        tp_groups: WorkerTPGroups,
        hostname: str = "localhost",
        socket_path_prefix: str = "/tmp/mstar",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
        device: torch.device = torch.device("cuda"),
        enable_nvtx: bool = False,
        enable_prof: bool = False,
        tcp_transfer_device="",
        dist_init_method=None
    ):
        self.worker_id = worker_id
        self.device = device
        self.enable_nvtx = enable_nvtx

        self.enable_prof = enable_prof
        self.profile_info = WorkerProfileInfo()

        # Fast path: send tiny integer new-token emit_to_client tensors inline
        # in the result_tensors message instead of via the SHM tensor
        # transport. Default OFF. See _inline_emit_uuids / _send_outputs.
        self._inline_emit = os.environ.get("MSTAR_INLINE_EMIT", "0") == "1"

        # Fast path: coalesce all qualifying inline emit_to_client messages of
        # one decode step (across every rid in the batch) into ONE
        # result_tensors_batch APIServerMessage, fanned out on the api_server
        # side. Default OFF. Batch implies inline: it is the single flag ruling
        # emission for qualifying edges, so enabling it turns on inline-emit
        # semantics for those edges even if MSTAR_INLINE_EMIT is not set.
        # Non-qualifying edges/messages are unaffected. See _send_outputs /
        # _postprocess_batch.
        self._batch_emit = os.environ.get("MSTAR_BATCH_EMIT", "0") == "1"
        if self._batch_emit:
            self._inline_emit = True

        # MSTAR_DIRECT_FEED: on uniform AR decode speculation (same-walk,
        # same-node loop-back), splice the spec batch's loop-back text_inputs
        # straight from batch_N's batched sampled-tokens tensor
        # (NodeOutput.batched_sampled_tokens) instead of the per-rid tensors
        # threaded out of the registry-backed output map. Default OFF; when
        # off _thread_outputs_to_speculative is byte-identical. The registry
        # store / route path (_postprocess_batch) is UNCHANGED either way — it
        # still populates the loop-back edge's ready-state + tensor_info that
        # the NEXT spec placeholder build (_get_input_tensors) and any fall
        # back to non-speculative decode depend on. See the block in
        # _thread_outputs_to_speculative for the exact safety conditions.
        self._direct_feed = os.environ.get("MSTAR_DIRECT_FEED", "0") == "1"

        # MSTAR_MULTISTEP_DECODE=N: run up to N thinker-decode steps inside ONE
        # GPU-thread submission, amortizing the per-step main-loop cycle
        # (await + speculate + thread + submit + prepare/plan/copy) over N
        # replays. Default 1 (OFF). Requires DIRECT_FEED: the between-step feed
        # of the sampled token into the next replay's static input buffers IS
        # the DIRECT_FEED mechanism, applied inline on the GPU thread. The
        # worker gates each submission (uniform thinker_decode, enough tokens
        # remaining) and postprocesses the returned steps SEQUENTIALLY as N
        # normal steps so Loop bookkeeping / emission / check_stop stay
        # per-token-exact. When off, the whole path is byte-identical. See
        # _eligible_multistep_decode / _postprocess_batch and
        # cuda_graph_runner._run_basic_batched.
        self._multistep_decode = max(
            1, int(os.environ.get("MSTAR_MULTISTEP_DECODE", "1"))
        )
        # Multistep is meaningless without DIRECT_FEED (it reuses that feed
        # mechanism between inline steps); force OFF if DIRECT_FEED is off so a
        # misconfiguration can't silently take an untested path.
        if self._multistep_decode > 1 and not self._direct_feed:
            logger.warning(
                "Worker %s: MSTAR_MULTISTEP_DECODE=%d requires "
                "MSTAR_DIRECT_FEED=1; disabling multistep.",
                self.worker_id, self._multistep_decode,
            )
            self._multistep_decode = 1

        if self.device.type == "cuda" and self.device.index is not None:
            torch.cuda.set_device(self.device)

        # ``dist_init_method`` is normally provided by the conductor — it
        # picks a free TCP port at startup so multiple ``mstar`` runs on
        # the same host don't collide. The ``tcp://{hostname}:29500``
        # fallback is for standalone Worker construction (e.g. tests);
        # production paths always pass a value.
        if dist_init_method is None:
            dist_init_method = f"tcp://{hostname}:29500"

        self.tp_groups = tp_groups
        self.tp_groups.init_dist(init_method=dist_init_method)

        # Build node_to_partition mapping from model's partitions and graph walks
        node_to_partition: dict[str, str] = {}
        if model is not None:
            partitions = model.get_partitions()
            walks = model.get_graph_walk_graphs()
            for pdef in partitions:
                for walk_name in pdef.graph_walks:
                    section = walks.get(walk_name)
                    if section:
                        for node_name in section.get_nodes():
                            node_to_partition[node_name] = pdef.name

        self.communicator = ZMQCommunicator(
            my_id=worker_id,
            push_ids=worker_ids + ["conductor", "api_server", "api_server_preprocess_worker"],
            ipc_socket_path_prefix=socket_path_prefix,
        )
        self.wakeup_event = EventWakeup()
        self.communicator.register_event_for_poll(self.wakeup_event)

        self.tensor_manager = create_tensor_communication_manager(
            protocol=tensor_comm_protocol,
            my_entity_id=worker_id,
            hostname=hostname,
            device=self.device,
            communicator=self.communicator,
            tcp_transfer_device=tcp_transfer_device,
            enable_prof=enable_prof
        )

        node_names = set()
        for wg in my_worker_graphs:
            node_names.update(wg.section.get_nodes())

        self.engine_manager = EngineManager.build(
            node_names,
            device=device,
            kv_config=kv_config,
            model_config=model_config,
            tp_groups=self.tp_groups,
            transfer_engine_info=TransferEngineInfo(
                my_entity_id=worker_id,
                my_session_id=self.tensor_manager.my_session_id,
                transfer_engine=self.tensor_manager.transfer_engine
            ),
            model=model,
            enable_nvtx=self.enable_nvtx,
            enable_prof=self.enable_prof
        )

        self.worker_graphs_manager = WorkerGraphsManager(
            queues={
                worker_graph.worker_graph_id: WorkerGraphQueues(
                    worker_graph_id=worker_graph.worker_graph_id,
                    graph_walks=worker_graph.graph_walks,
                    worker_graph=worker_graph,
                    per_request_queues={},
                    tensor_manager=self.tensor_manager
                )
                for worker_graph in my_worker_graphs
            },
            per_request_info={},
            all_worker_graph_ids_to_graph_walks=all_worker_graph_ids_to_graph_walks,
            all_worker_graph_ids_to_dyn_loops=all_worker_graph_ids_to_dyn_loops,
            all_worker_graph_ids_to_nodes=all_worker_graph_ids_to_nodes,
            node_to_partition=node_to_partition,
            base_sharding_config=sharding_config,
            worker_id=self.worker_id
        )

        self.tp_rank_zero_nodes = set([
            node for node in node_names if self.tp_groups.get_tp_config_for_node(node).rank == 0
        ])

        # v1: disallow multiple TP nodes in the same worker
        self.tp_nodes = set([
            node for node in node_names if self.tp_groups.get_tp_config_for_node(node).world_size > 1
        ])
        if len(self.tp_nodes) > 1:
            raise NotImplementedError(
                f"Multiple TP nodes {self.tp_nodes} found in worker {worker_id}; "
                "current implementation requires at most one TP node per worker."
            )

        self.is_tp_follower = len(self.tp_nodes - self.tp_rank_zero_nodes) > 0

        self.scheduler = MicroScheduler(
            self.engine_manager,
            tp_rank_zero_nodes=self.tp_rank_zero_nodes
        )

        # Determine store write policy based on worker graph topology
        node_engine_types = model.get_node_engine_types() if model is not None else {}
        write_policy = self._compute_store_write_policy(
            my_worker_graphs, all_worker_graph_ids_to_graph_walks,
            all_worker_graph_ids_to_nodes,
            node_engine_types=node_engine_types,
        )
        self.engine_manager.set_alloc_write_policies(write_policy)
        logger.info(
            "Worker %s: store write policy = %s", worker_id, write_policy.value
        )

        self._unprocessed_messages = {} # req_id -> messages for requests that are not in the queue

        # CPU offloading: LRU tracking and eviction policy
        self._last_active: dict[tuple[str, str], float] = {}  # (request_id, node_name) -> monotonic timestamp
        self.eviction_policy = EvictionPolicy.LRU

        # Async-scheduling cross-iter state. Initialized here (rather than in
        # run()) because _remove_request — which can be invoked indirectly
        # from _process_messages on any iter — reads/writes them.
        # _in_flight_rids: rids referenced by an in-flight GPU step or its
        #   speculation; REMOVE_REQUEST for these is deferred.
        # _pending_removes: deferred REMOVE_REQUESTs.
        # _pending_loop_stops: loop-stops produced by check_stop in this iter's
        #   postprocess, consumed by next iter's speculation to drop rids whose
        #   loop has ended. Keyed by (rid, graph_walk, loop_name) — see
        #   PendingLoopStop.
        self._in_flight_rids: set[str] = set()
        self._pending_removes: set[str] = set()
        self._pending_loop_stops: set[PendingLoopStop] = set()
        # Let the scheduler see deferred removes so it stops initiating new work
        # for those rids (shared by reference — mutations are visible to both).
        self.scheduler.pending_removes = self._pending_removes

        # Side stream for D→H copies in postprocess (check_stop pre-materialize).
        # The default stream has GPU(N+1) queued behind GPU(N)'s outputs after
        # speculation, so syncing on default would also drain GPU(N+1) and
        # erase the overlap. The side stream waits on
        # ``output.completion_event`` (recorded after GPU(N)) and then runs
        # an isolated D→H, so the main thread only blocks on the copy.
        # Lazy-initialized — workers without CUDA never touch it.
        self._d2h_stream: "torch.cuda.Stream | None" = None
        self._pinned_d2h_buffers: dict[
            tuple[str, torch.dtype, tuple[int, ...]], list[torch.Tensor]
        ] = defaultdict(list)

        # MSTAR_SIDE_PREFILL: run a thinker-prefill (or stateless encoder)
        # batch CONCURRENTLY with the decode chain on a second CUDA stream +
        # second executor thread, instead of yielding the decode chain to run
        # it inline. Default OFF. See run() for the loop restructure and
        # _execute_on_gpu_thread for the stream plumbing.
        #
        # Correctness note on KV: the thinker-prefill hands its first token to
        # decode ASYNCHRONOUSLY through the conductor (persist → conductor
        # rebuilds text_inputs → back as a fresh decode batch a later iter),
        # NOT via a local graph edge. The side batch's postprocess runs on the
        # main thread and fully drains the side stream (the completion_event is
        # recorded ON the side stream, and _prematerialize_for_check_stop
        # side.wait_event(completion_event)+synchronize before the token is
        # even routed). So the prefill's KV pages are durably written long
        # before the decode step for that rid arrives — no cross-stream
        # wait_event is needed on the decode replay path. The single invariant
        # is: record the side batch's completion_event on the side stream (see
        # _execute_on_gpu_thread), so the existing token-materialization wait
        # gates on the right stream.
        self._side_prefill = os.environ.get("MSTAR_SIDE_PREFILL", "0") == "1"
        self._side_stream: "torch.cuda.Stream | None" = None
        # rids currently executing on the side stream — treated as in-flight
        # for deferred-remove safety (see _apply_pending_removes_safe_to_drop).
        self._side_in_flight_rids: set[str] = set()
        # Belt-and-suspenders: the scheduler is single-caller by construction
        # (main thread owns all get_next_batch / has_ready_excluding calls; the
        # side executor only EXECUTES pre-built batches). This lock guards the
        # scan+pop critical section so the invariant is defended even if a
        # future change slips a scheduler call onto another thread.
        self._scheduler_lock = threading.Lock()

        # Streaming buffers: request_id -> edge_name -> list of tensors
        # (Legacy path — kept for models without PartitionTopology)
        self.streaming_buffers: dict[str, dict[str, list[torch.Tensor]]] = {}

        # New streaming path: PartitionTopology + StreamBuffer on consumer worker
        self.partition_topology = model.get_partition_topology() if model else None

        # Determine which partition this worker serves (by checking which node names
        # appear in my_worker_graphs vs the topology connections)
        self._my_consumer_connections = []
        if self.partition_topology:
            my_node_names = set()
            for wg in my_worker_graphs:
                my_node_names.update(wg.section.get_nodes())
            for conn in self.partition_topology.connections:
                # Check if any graph walk graph node for the consumer partition is on this worker
                # by checking if the streaming edge's next_node is in my nodes
                if any(n in my_node_names for n in self._get_node_names_for_partition(conn.to_partition, model)):
                    self._my_consumer_connections.append(conn)

        # Set of edge names that arrive via streaming (used to distinguish
        # streaming inputs from conductor-triggered non-streaming inputs
        # when checking whether a target node is ready for ingestion).
        self._streaming_edge_names: set[str] = {
            conn.edge_name for conn in self._my_consumer_connections
        }

        # Build consumer node cache: edge_name -> next_node name
        self._consumer_node_cache: dict[str, str] = {}
        if self._my_consumer_connections and model:
            walks = model.get_graph_walk_graphs()
            for conn in self._my_consumer_connections:
                for section in walks.values():
                    if hasattr(section, 'input_names') and conn.edge_name in section.input_names:
                        self._consumer_node_cache[conn.edge_name] = section.name

    def _get_node_names_for_partition(self, partition_name: str, model: Model) -> list[str]:
        """Get the node names that belong to a partition."""
        walks = model.get_graph_walk_graphs()
        partitions = model.get_partitions()
        for pdef in partitions:
            if pdef.name == partition_name:
                nodes = set()
                for walk_name in pdef.graph_walks:
                    section = walks.get(walk_name)
                    if section and hasattr(section, 'name'):
                        nodes.add(section.name)
                return list(nodes)
        return []

    def _compute_store_write_policy(
        self,
        my_worker_graphs: list[WorkerGraph],
        all_worker_graph_ids_to_graph_walks: dict[str, set[str]],
        all_worker_graph_ids_to_nodes: dict[str, set[str]],
        node_engine_types: dict[str, EngineType] | None = None,
    ) -> StoreWritePolicy:
        """Determine whether this worker needs to write KV to the mooncake store.

        If this worker handles ALL AR engine graph walks, no other worker
        needs its KV cache — return NEVER. Otherwise return ALWAYS.
        """
        my_ar_walks_nodes: set[str] = set()
        all_ar_walks_nodes: set[str] = set()

        def _uses_kv_cache(node_name: str) -> bool:
            # Local engine instance wins via declared capability; remote
            # nodes fall back to the model's static type map.
            engine = self.engine_manager.node_to_engine.get(node_name)
            if engine is not None:
                return engine.capabilities.requires_kv_cache
            if node_engine_types and node_name in node_engine_types:
                return node_engine_types[node_name] == EngineType.KV_CACHE
            return False

        # Collect this worker's AR graph walks
        for wg in my_worker_graphs:
            for node_name in wg.section.get_nodes():
                if _uses_kv_cache(node_name):
                    my_ar_walks_nodes.update([(walk, node_name) for walk in wg.graph_walks])

        # Collect all workers' AR graph walks
        for wg_id, walks in all_worker_graph_ids_to_graph_walks.items():
            nodes = all_worker_graph_ids_to_nodes.get(wg_id, set())
            for node_name in nodes:
                if _uses_kv_cache(node_name):
                    all_ar_walks_nodes.update([(walk, node_name) for walk in walks])

        if not all_ar_walks_nodes:
            return StoreWritePolicy.NEVER  # no AR engines at all

        if my_ar_walks_nodes == all_ar_walks_nodes:
            logger.info(
                "No LLM disaggregation detected; my_ar_walks_nodes == all_ar_walks_nodes: %s",
                str(my_ar_walks_nodes)
            )
            return StoreWritePolicy.NEVER  # all AR walks on this worker

        return StoreWritePolicy.ALWAYS

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _add_new_request(self, body: NewRequest) -> None:
        logger.debug("Worker %s received request %s", self.worker_id, body.request_id)
        now = _time.monotonic()
        for node_name in self.engine_manager.lru_tracked_nodes():
            self._last_active[(body.request_id, node_name)] = now

        self.worker_graphs_manager.add_request(
            request_id=body.request_id,
            partition_worker_graph_ids=body.partition_worker_graph_ids,
            worker_graph_to_workers=body.worker_graph_to_workers,
            current_fwd_info=body.request_info
        )
        self.engine_manager.add_request(body.request_id)
        self.tensor_manager.register_request(
            body.request_id,
            self.worker_graphs_manager.per_request_info[body.request_id].sharding_config
        )

        # Create StreamBuffers for consumer connections on this worker
        for conn in self._my_consumer_connections:
            req_info = self.worker_graphs_manager.per_request_info[body.request_id]
            req_info.stream_buffers[conn.edge_name] = StreamBuffer(
                request_id=body.request_id,
                edge_name=conn.edge_name,
                from_partition=conn.from_partition,
                policy=conn.chunk_policy_factory(),
            )

        # Start RDMA reads for tensors that have tensor_info
        futures = self.tensor_manager.start_read_tensors(
            body.request_id, body.initial_inputs,
            graph_walk=body.request_info.graph_walk
        )
        self.wakeup_event.register_futures(futures)

        # Signal-only edges (tensor_info is None) can be processed immediately
        signal_only = [
            edge for edge in body.initial_inputs if len(edge.tensor_info) == 0
        ]
        if signal_only:
            self.worker_graphs_manager.process_new_inputs(
                request_id=body.request_id, inputs=signal_only,
                can_buffer=True
            )
        # process messages that may have came in out-of-order
        if body.request_id in self._unprocessed_messages:
            self._process_message_list(self._unprocessed_messages[body.request_id])
            del self._unprocessed_messages[body.request_id]


    def _remove_request(self, body: RemoveRequest) -> None:
        if self.is_tp_follower and body.source not in (MessageSource.TP_RANK_0, MessageSource.SELF):
            return # wait for removal message from TP rank 0 to avoid race conditions

        # Async-scheduling deferral: if this rid is currently held by an
        # in-flight GPU step (or its speculation), tearing down engine /
        # tensor state now would race the GPU thread reading those tensors
        # / KV pages. Queue the remove and apply it once no in-flight step
        # references the rid (see _apply_pending_removes_safe_to_drop in
        # the run loop).
        if body.request_id in getattr(self, "_in_flight_rids", set()):
            self._pending_removes.add(body.request_id)
            return

        # If we are the TP leader for this request, signal the followers to
        # remove it too. Followers defer removal until they get this message
        # (see the guard at the top of this method) so they can't tear down
        # state we're still reading from an in-flight step/speculation.
        cfg = self.worker_graphs_manager.per_request_info.get(body.request_id)
        if cfg is not None:
            followers: set[str] = set()
            for group in cfg.sharding_config.groups:
                # _workers is rank-ordered; index 0 is this worker when we are
                # rank 0. Only real TP groups (tp_size > 1) have followers.
                if group.tp_size > 1 and group._tp_rank == 0:
                    followers.update(group._workers[1:])
            for worker in followers:
                self.communicator.send(
                    worker, msg=WorkerMessage(
                        message_type=WorkerMessageType.REMOVE_REQUEST,
                        body=RemoveRequest(
                            request_id=body.request_id,
                            source=MessageSource.TP_RANK_0,
                        )
                    )
                )

        self.engine_manager.remove_request(body.request_id)
        self.worker_graphs_manager.remove_request(body.request_id)
        self.tensor_manager.cleanup_request(body.request_id)
        self.profile_info.pop_request(body.request_id)
        self.streaming_buffers.pop(body.request_id, None)

        for node_name in self.engine_manager.lru_tracked_nodes():
            self._last_active.pop((body.request_id, node_name), None)

    def _handle_tensor_received(self, body: TensorReceived) -> None:
        """Sender-side cleanup: receiver confirmed RDMA read, free source buffers."""
        for (uuid, ref_cnt) in body.successful_tensors.items():
            self.tensor_manager.dereference(
                body.request_id, uuid, n=ref_cnt
            )

    def _process_new_inputs(self, body: InputSignals) -> None:
        logger.debug(
            "Received new signals %s at worker %s for request %s",
            format_graph_edge_list(body.inputs), self.worker_id, body.request_id
        )
        req_info = self.worker_graphs_manager.per_request_info.get(body.request_id)

        if self.enable_nvtx:
            range_push("process_new_inputs.routing_update")
        # Handle producer_done signal: mark all StreamBuffers for this request as done
        if body.producer_done:
            if req_info:
                for sbuf in req_info.stream_buffers.values():
                    if sbuf.from_partition in body.producer_done:
                        # If we have multiple consumer partitions colocated, we need to signal
                        # the right one
                        sbuf.signal_done()

        # Separate streaming edges — they'll be handled when tensors are ready
        # (streaming edges with tensor_info go through RDMA, handled in _check_ready_tensors)
        non_streaming = [edge for edge in body.inputs if not edge.is_streaming]
        streaming_with_tensors = [edge for edge in body.inputs if edge.is_streaming and edge.tensor_info]

        # Only update fwd_info when there are non-streaming edges (i.e., this is
        # a conductor-triggered forward pass, not just streaming data from another
        # partition). Streaming-only InputSignals must not overwrite the current
        # partition's fwd_info.
        if non_streaming:
            self.worker_graphs_manager.update_request_info(
                body.request_id, current_fwd_info=body.request_info,
                partition_name=body.partition_name
            )

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("process_new_inputs.start_read")
        # Start RDMA reads for non-streaming edges with tensor_info
        futures = self.tensor_manager.start_read_tensors(
            body.request_id, non_streaming,
            graph_walk=body.request_info.graph_walk
        )
        self.wakeup_event.register_futures(futures)
        # Start RDMA reads for streaming edges with tensor_info (will be routed to buffer in _check_ready_tensors)
        if streaming_with_tensors:
            futures = self.tensor_manager.start_read_tensors(
                body.request_id, streaming_with_tensors,
            )
            self.wakeup_event.register_futures(futures)
            for edge in streaming_with_tensors:
                stream_buf = req_info.stream_buffers[edge.name]
                for info in edge.tensor_info:
                    stream_buf.pre_read_register(info.uuid)
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("process_new_inputs.process_inputs")

        # Streaming signal-only edges: nothing to buffer (no tensor data)
        # This shouldn't normally happen for streaming edges

        # Signal-only non-streaming edges can be processed immediately
        signal_only = [edge for edge in non_streaming if len(edge.tensor_info) == 0]
        if signal_only:
            self.worker_graphs_manager.process_new_inputs(
                request_id=body.request_id, inputs=signal_only,
                can_buffer=True
            )
        if self.enable_nvtx:
            range_pop()

    def _unpersist_tensors(self, body: UnpersistTensors):
        for (uuid, ref_cnt) in body.uuid_to_ref_count.items():
            self.tensor_manager.increment_ref(
                body.request_id, uuid, n=ref_cnt
            )
            self.tensor_manager.set_persist(
                body.request_id, uuid, persist=False
            )

    def _stop_loops(self, body: StopLoops):
        if not self.worker_graphs_manager.has_partition(
            body.request_id, body.partition_name
        ):
            return
        fwd_info = self.worker_graphs_manager.get_fwd_info(
            body.request_id, body.partition_name
        )
        loop_names = set()
        for name, stop_time in body.loop_stop_times.items():
            if name not in fwd_info.loop_stop_times or stop_time.label_context_gt(
                fwd_info.loop_stop_times[name], name
            ):
                loop_names.add(name)
            fwd_info.loop_stop_times[name] = stop_time
        if loop_names:
            self.worker_graphs_manager.stop_loops(
                body.request_id, body.partition_name, loop_names
            )

    def _process_message_list(self, messages: list[WorkerMessage]):
        msg_types_needing_active_request = [
            WorkerMessageType.REMOVE_REQUEST,
            WorkerMessageType.INPUT_SIGNALS,
            WorkerMessageType.STOP_LOOPS
        ]
        # Snapshot: a REMOVE handled mid-iteration can re-buffer trailing
        # signals onto this same list, and mutating it while iterating it would
        # never terminate.
        for message in list(messages):
            if (
                message.message_type in msg_types_needing_active_request and \
                message.body.request_id not in self.worker_graphs_manager.per_request_info
            ):
                # got an out-of-order request
                self._unprocessed_messages.setdefault(
                    message.body.request_id, []
                ).append(message)
                continue
            if message.message_type == WorkerMessageType.NEW_REQUEST:
                self._add_new_request(message.body)
            elif message.message_type == WorkerMessageType.REMOVE_REQUEST:
                self._remove_request(message.body)
            elif message.message_type == WorkerMessageType.INPUT_SIGNALS:
                self._process_new_inputs(message.body)
            elif message.message_type == WorkerMessageType.TENSOR_RECEIVED:
                self._handle_tensor_received(message.body)
            elif message.message_type == WorkerMessageType.UNPERSIST_TENSORS:
                self._unpersist_tensors(message.body)
            elif message.message_type == WorkerMessageType.STOP_LOOPS:
                self._stop_loops(message.body)
            elif message.message_type == WorkerMessageType.SCHEDULE_TP:
                self.scheduler.register_tp_follow(message.body)

    def _process_messages(self) -> None:
        self._process_message_list(self.communicator.get_all_new_messages())

    # ------------------------------------------------------------------
    # Tensor readiness
    # ------------------------------------------------------------------

    def _route_streaming_tensor(self, request_id: str, edge: GraphEdge) -> None:
        """Route a streaming tensor to its request's StreamBuffer for this edge."""
        req_info = self.worker_graphs_manager.per_request_info.get(request_id)
        stream_buf = req_info.stream_buffers[edge.name]

        for info in edge.tensor_info:
            tensor = self.tensor_manager.get_tensor(
                request_id=request_id, uuid=info.uuid,
            )

            stream_buf.put(info.uuid, tensor.clone())
            self.tensor_manager.dereference(request_id, info.uuid)

    def _pop_streaming_edge(
        self, sbuf: StreamBuffer, edge_name: str, request_id: str
    ) -> GraphEdge | None:
        consumer_node = self._consumer_node_cache.get(edge_name, "")
        synthetic_edge = sbuf.pop_waiting_edge()
        if synthetic_edge is None and sbuf.has_chunk_ready():
            chunk = sbuf.pop_chunk()
            chunk_tensor = chunk.data.get("data")
            if chunk_tensor is None:
                # Empty chunk — producer done, no more data.
                # Create edge with empty tensor_info.
                synthetic_edge = GraphEdge(
                    next_node=consumer_node,
                    name=edge_name,
                    tensor_info=[],
                    _final_stream_chunk=chunk.is_final,
                )
            else:
                # Normal chunk — store tensor and create edge with tensor_info.
                # Local streaming tensors are routed from outputs that were
                # already gated on the producer completion event before being
                # stored, so avoid a default-stream sync here. If future
                # streaming producers bypass that path, StreamChunk should
                # carry producer events and this call site should wait on
                # those events before storing with skip_cuda_sync=True.
                tensor_infos = self.tensor_manager.store_and_return_tensor_info(
                    request_id, {edge_name: [chunk_tensor]},
                    skip_cuda_sync=True,
                )
                synthetic_edge = GraphEdge(
                    next_node=consumer_node,
                    name=edge_name,
                    tensor_info=tensor_infos.get(edge_name, []),
                    _final_stream_chunk=chunk.is_final,
                )
        return synthetic_edge

    def _poll_stream_buffers_for_speculation(
        self, request_id: str, node_name: str
    ) -> list[GraphEdge]:
        result = []
        req_info = self.worker_graphs_manager.per_request_info.get(request_id)
        if req_info is None:
            return []
        for edge_name, sbuf in req_info.stream_buffers.items():
            consumer_node = self._consumer_node_cache.get(edge_name, "")
            if consumer_node != node_name:
                continue
            edge = self._pop_streaming_edge(sbuf, edge_name, request_id)
            if edge is not None:
                result.append(edge)
        return result

    def _return_speculative_streaming_edge(
        self, request_id: str, edge: GraphEdge
    ):
        req_info = self.worker_graphs_manager.per_request_info.get(request_id)
        if req_info is None:
            return
        sbuf = req_info.stream_buffers.get(edge.name)
        if sbuf is not None:
            sbuf.store_uningested_edge(edge)

    def _poll_stream_buffers(self) -> None:
        """Check all active StreamBuffers; when a chunk is ready, feed it as a normal input."""
        for request_id, req_info in list(self.worker_graphs_manager.per_request_info.items()):
            for edge_name, sbuf in req_info.stream_buffers.items():
                synthetic_edge = self._pop_streaming_edge(sbuf, edge_name, request_id)

                if synthetic_edge is not None:
                    # Streaming edges go through the same path as regular ones —
                    # ReadySignals.is_ready_for_streaming flips on as soon as
                    # the streaming inputs are the only ones missing. Empty
                    # leftover list means the edge was claimed. The final-chunk
                    # signal rides the synthetic edge to the consuming pass,
                    # which reports the partition done in _postprocess_batch —
                    # NOT here, where an earlier in-flight pass's WGD could read
                    # it before the final output chunk is emitted.
                    leftovers = self.worker_graphs_manager.process_new_streaming_inputs(
                        request_id=request_id, inputs=[synthetic_edge],
                        can_buffer=False # important: only ingest for this loop iter only!
                    )
                    if leftovers:
                        sbuf.store_uningested_edge(synthetic_edge)


    def _check_ready_tensors(self) -> None:
        """Poll for completed RDMA transfers, feed ready graph edges to worker graph queues."""
        self.wakeup_event.drain()
        ready = self.tensor_manager.get_ready_tensors()
        for request_id, edges in ready.items():
            # Separate streaming edges from normal edges
            streaming = [e for e in edges if e.is_streaming]
            normal = [e for e in edges if not e.is_streaming]

            if self.enable_nvtx:
                range_push("check_ready-tensors.route_streaming")
            for edge in streaming:
                self._route_streaming_tensor(request_id, edge)

            if self.enable_nvtx:
                range_pop(synchronize=False)
                range_push("process_new_inputs.process_inputs")

            if normal:
                self.worker_graphs_manager.process_new_inputs(
                    request_id=request_id, inputs=normal,
                    can_buffer=True
                )
            if self.enable_nvtx:
                range_pop(synchronize=False)

    # ------------------------------------------------------------------
    # CPU offloading
    # ------------------------------------------------------------------

    def _try_offload_cold_request(
        self, node_name: str, batch_ids: set[str]
    ) -> str | None:
        """Offload one request's KV pages to CPU using the configured eviction policy.

        Prefers requests outside *batch_ids*. If none exist, falls back to
        picking a victim *within* the batch (the caller should then exclude
        it from execution).

        Returns the victim request_id, or None if offloading wasn't possible.
        """
        engine = self.engine_manager.get_engine(node_name)
        if not engine.capabilities.supports_cpu_offload:
            return None

        candidates_raw = engine.offload_candidates(node_name)
        if not candidates_raw:
            return None

        # Split candidates by whether they belong to the in-flight batch;
        # we prefer evicting requests not currently being executed.
        external: list[tuple[str, int]] = []
        in_batch: list[tuple[str, int]] = []
        for rid, total_pages in candidates_raw:
            if rid in batch_ids:
                in_batch.append((rid, total_pages))
            else:
                external.append((rid, total_pages))

        candidates = external or in_batch
        if not candidates:
            return None

        victim_id = self._select_eviction_victim(node_name, candidates)
        freed = engine.offload_request(node_name, victim_id)
        logger.info(
            "Offloaded request %s to CPU (%d GPU pages freed, "
            "policy=%s, in_batch=%s)",
            victim_id, freed, self.eviction_policy.value,
            victim_id in batch_ids,
        )
        return victim_id if freed > 0 else None

    def _select_eviction_victim(
        self, node_name: str, candidates: list[tuple[str, int]]
    ) -> str:
        """Pick a victim from *candidates* based on ``self.eviction_policy``.

        Each candidate is ``(request_id, total_gpu_pages)``.
        """
        if self.eviction_policy == EvictionPolicy.MOST_PAGES:
            return max(candidates, key=lambda x: x[1])[0]

        # LRU: pick the request with the oldest last_active timestamp.
        # Ties (or missing entries) broken by most pages.
        return min(
            candidates,
            key=lambda x: (
                self._last_active.get((x[0], node_name), 0.0),  # oldest first
                -x[1],                               # then most pages
            ),
        )[0]

    def _try_reload_request(self, node_name: str, request_id: str) -> bool:
        """Reload an offloaded request back to GPU. Returns True if reloaded."""
        engine = self.engine_manager.get_engine(node_name)
        if not engine.is_offloaded(node_name, request_id):
            return False
        if engine.reload_request(node_name, request_id):
            logger.info("Reloaded request %s from CPU to GPU", request_id)
            return True
        logger.debug(
            "Cannot reload request %s yet (insufficient GPU pages)", request_id,
        )
        return False

    # ------------------------------------------------------------------
    # Batch building
    # ------------------------------------------------------------------

    def _build_node_batch(self, batch: ScheduledBatch) -> NodeBatch:
        """Gather input tensors from tensor_manager for all requests in the batch."""
        per_request_inputs: dict[str, NameToTensorList] = {}
        per_request_info: dict[CurrentForwardPassInfo] = {}
        final_stream_rids: set[str] = set()
        batch_partition = self.worker_graphs_manager.get_partition_for_node(batch.node_name)

        for request_id, node in batch.node_objects.items():
            tensors = {}
            ready_inputs = node.ready_signals.ready_inputs
            for input_name, edge in ready_inputs.items():
                tensors[input_name] = [
                    self.tensor_manager.get_tensor(
                        request_id=request_id, uuid=info.uuid
                    ) for info in edge.tensor_info
                ]
                if edge._final_stream_chunk:
                    final_stream_rids.add(request_id)
            per_request_inputs[request_id] = tensors
            per_request_info[request_id] = self.worker_graphs_manager.get_fwd_info(request_id, batch_partition)

        return NodeBatch(
            node_name=batch.node_name,
            graph_walk=batch.graph_walk,
            request_ids=list(batch.node_objects.keys()),
            per_request_input_tensors=per_request_inputs,
            per_request_info=per_request_info,
            final_stream_rids=final_stream_rids,
        )

    def maybe_send_zmq_to_tp_followers(
        self, node_batch: NodeBatch
    ):
        if node_batch.node_name not in self.tp_nodes or \
                node_batch.node_name not in self.tp_rank_zero_nodes:
            return
        # this worker is only a part of one TP group for this node,
        # so, we can just look at the sharding_config for the first
        # request to get the relevant workers
        sample_rid = node_batch.request_ids[0]
        cfg = self.worker_graphs_manager.per_request_info[sample_rid]
        workers = cfg.sharding_config.get_sharding_group(
            node_batch.node_name, node_batch.graph_walk
        )._workers[1:]
        for worker in workers:
            self.communicator.send(
                worker, msg=WorkerMessage(
                    message_type=WorkerMessageType.SCHEDULE_TP,
                    body=ScheduleTPNode(
                        node_name=node_batch.node_name,
                        graph_walk=node_batch.graph_walk,
                        request_ids=node_batch.request_ids
                    )
                )
            )

    # ------------------------------------------------------------------
    # Output handling
    # ------------------------------------------------------------------
    def _inline_emit_uuids(
        self,
        routing: NodeOutputRouting,
        prematerialized_new_tokens: dict[str, list[int]] | None,
    ) -> set[str]:
        """UUIDs of emit_to_client tensors eligible for the inline fast path.

        A uuid qualifies only when (a) the inline-emit flag is on, (b) its
        edge is a tiny integer new-token tensor already prematerialized to
        CPU ints (present in ``prematerialized_new_tokens``), and (c) the
        uuid is used ONLY by qualifying emit_to_client edges. Condition (c)
        is essential: the same produced tensor's uuid can also feed a
        persist / loop-back (to_workers) / streaming edge, which still need
        the SHM transport — skipping their SHM write would break the
        consumer's read. So we exclude any uuid that appears on a
        non-inline edge.
        """
        if not self._inline_emit or not prematerialized_new_tokens:
            return set()

        inline_candidates: set[str] = set()
        for edge in routing.emit_to_client:
            if edge.name not in prematerialized_new_tokens:
                continue
            # Only integer new-token edges are prematerialized; audio /
            # multimodal edges are never in the prem dict, so they never
            # reach here.
            inline_candidates.update(info.uuid for info in edge.tensor_info)

        if not inline_candidates:
            return set()

        # Any uuid also referenced by a non-inline consumer must keep its
        # SHM write; drop it from the inline set.
        non_inline_uuids: set[str] = set()
        for edge in (
            routing.persist +
            sum(routing.to_workers.values(), start=[]) +
            sum(routing.streaming_to_workers.values(), start=[]) +
            routing.streaming_local +
            routing.routed_to_this_worker_graph
        ):
            non_inline_uuids.update(info.uuid for info in edge.tensor_info)

        return inline_candidates - non_inline_uuids

    def _register_outputs(
        self,
        batch: ScheduledBatch,
        routing_per_request: dict[str, NodeOutputRouting],
        prematerialized_per_request: dict[str, dict[str, list[int]] | None] | None = None,
    ):
        """
        For outputs going to other workers: register tensors for RDMA send
        and populate tensor_info on the GraphEdges.
        For outputs staying local: store tensors in tensor_manager.
        Returns the output edges per request (with tensor_info filled in).

        ``prematerialized_per_request`` (optional): per-rid prematerialized
        new-token ints, used to identify emit_to_client uuids that will be
        sent inline (see ``_inline_emit_uuids``). Inline uuids are NOT
        registered for send — no SHM file is written for them.
        """
        for request_id, _node in batch.node_objects.items():
            routing = routing_per_request[request_id]
            prem = (
                (prematerialized_per_request or {}).get(request_id)
            )
            inline_uuids = self._inline_emit_uuids(routing, prem)
            uuids = set()
            for edge in (
                routing.persist +
                sum(routing.to_workers.values(), start=[]) +
                routing.emit_to_client +
                sum(routing.streaming_to_workers.values(), start=[])
            ):
                uuids.update([
                    info.uuid for info in edge.tensor_info
                ])
            # Inline-emit uuids skip SHM registration entirely: no file
            # write, no remote fetch, no ack. Their producer-side ref is
            # released locally in _send_outputs instead.
            uuids -= inline_uuids
            self.tensor_manager.register_for_send(
                request_id=request_id, uuids=uuids,
                skip_cuda_sync=True,
            )

            for edge in routing.persist:
                for info in edge.tensor_info:
                    self.tensor_manager.set_persist(
                        request_id=request_id, uuid=info.uuid, persist=True
                    )


    def _send_outputs(
        self, request_id: str, outputs: NodeOutputRouting,
        nested_loop_indices: NestedLoopIndices,
        graph_walk: str | None = None,
        partition_name: str | None = None,
        prematerialized_new_tokens: dict[str, list[int]] | None = None,
        node_speculatively_scheduled: bool=False,
        batch_collector: list["ResultTensors"] | None = None,
    ) -> None:
        """
        Send outputs to other workers and to the conductor.
        Persist signals are buffered and sent together with the
        WORKER_GRAPHS_DONE message to avoid race conditions.

        ``prematerialized_new_tokens`` (optional): `{signal_name: [int, ...]}`
        for this request, where the caller has already done the D→H copy
        for the new-token tensors. When provided, this function skips the
        per-tensor ``.cpu()`` call — meaningful when the caller batched
        multiple requests' new-token transfers into a single D→H to avoid
        N serialized ``cudaMemcpyAsync`` + ``cudaStreamSynchronize`` per
        step.

        ``batch_collector`` (optional): when supplied (MSTAR_BATCH_EMIT), a
        qualifying inline emit_to_client edge's ``ResultTensors`` is appended
        to this list instead of being sent as its own result_tensors message;
        the caller coalesces the whole step's collector into a single
        result_tensors_batch message. ONLY inline-qualifying edges are
        collected — every non-inline emit edge (and every other message here)
        is sent immediately exactly as without the flag. The producer-side
        ref release for inline uuids is unchanged: it happens here per rid.
        """
        if graph_walk is None:
            graph_walk = self.worker_graphs_manager.get_graph_walk(request_id, partition_name)
        for worker_id, edges in outputs.to_workers.items():
            message = WorkerMessage(
                message_type=WorkerMessageType.INPUT_SIGNALS,
                body=InputSignals(
                    request_id=request_id,
                    inputs=edges,
                    request_info=self.worker_graphs_manager.get_fwd_info(request_id, partition_name),
                    partition_name=partition_name
                ),
            )
            self.communicator.send(worker_id, message)

        # Buffer persist signals for this request
        if outputs.persist:
            self.worker_graphs_manager.buffer_persist_signals(
                request_id, outputs.persist
            )

        if outputs.new_token_outputs:
            name_to_new_token: dict = {}
            for signal in outputs.new_token_outputs:
                if signal.name in name_to_new_token:
                    continue # don't double-count new tokens
                if (
                    prematerialized_new_tokens is not None
                    and signal.name in prematerialized_new_tokens
                ):
                    new_tokens = prematerialized_new_tokens[signal.name]
                else:
                    new_tokens = []  # list[int]
                    for tensor_info in signal.tensor_info:
                        tensor = self.tensor_manager.get_tensor(
                            request_id=request_id,
                            uuid=tensor_info.uuid
                        )
                        new_tokens.extend(tensor.cpu().numpy().tolist())
                name_to_new_token[signal.name] = new_tokens

                self.worker_graphs_manager.buffer_new_tokens(
                    request_id, name_to_new_token
                )

        if outputs.emit_to_client:
            self.worker_graphs_manager.buffer_output_signals(
                request_id, outputs.emit_to_client
            )
            inline_uuids = self._inline_emit_uuids(
                outputs, prematerialized_new_tokens
            )
            # uuids we release locally, weighted by how many emit tensor_info
            # entries reference each (mirrors the per-tensor_info ack count
            # the data worker would have sent via TENSOR_RECEIVED).
            local_release: dict[str, int] = {}
            for graph_edge in outputs.emit_to_client:
                self.worker_graphs_manager.register_output_loop_indices(
                    request_id=request_id, loop_indices=nested_loop_indices,
                    output_name=graph_edge.name
                )
                metadata: dict = {}
                edge_inline = self._inline_emit and bool(graph_edge.tensor_info) and all(
                    info.uuid in inline_uuids for info in graph_edge.tensor_info
                )
                if edge_inline:
                    # Carry the token values inline; the consumer skips the
                    # SHM fetch entirely. dtype/shape come from tensor_info
                    # on the (still-attached) graph_edge, so the consumer
                    # reconstructs a byte-identical tensor for postprocess.
                    metadata = {
                        "inline_values": {
                            graph_edge.name: prematerialized_new_tokens[graph_edge.name]
                        }
                    }
                    for info in graph_edge.tensor_info:
                        local_release[info.uuid] = local_release.get(info.uuid, 0) + 1
                result_tensors = ResultTensors(
                    request_id=request_id,
                    modality=graph_edge.output_modality,
                    graph_edge=graph_edge,
                    loop_indices=nested_loop_indices,
                    metadata=metadata
                )
                if batch_collector is not None and edge_inline:
                    # Coalesced path: defer to a single result_tensors_batch
                    # message built by the caller after the rid loop. Only
                    # inline edges are collected — non-inline edges below still
                    # send their own message, byte-identical to the flag-off
                    # path.
                    batch_collector.append(result_tensors)
                else:
                    message = APIServerMessage(
                        message_type="result_tensors",
                        body=result_tensors,
                    )
                    self.communicator.send("api_server", message)

            # Release the producer-side ref for inline uuids now: no
            # TENSOR_RECEIVED ack will ever arrive for them (they were never
            # registered for send in _register_outputs), so this stands in
            # for the ack's dereference and prevents a tensor_store leak.
            for uuid, n in local_release.items():
                self.tensor_manager.dereference(request_id, uuid, n=n)

        # Handle streaming edges
        # Local streaming: route to StreamBuffer
        req_info = self.worker_graphs_manager.per_request_info[request_id]
        for edge in outputs.streaming_local:
            stream_buf = req_info.stream_buffers[edge.name]
            for info in edge.tensor_info:
                stream_buf.pre_read_register(info.uuid)
            self._route_streaming_tensor(request_id, edge)

        # Remote streaming: send to destination workers
        for worker_id, edges in outputs.streaming_to_workers.items():
            message = WorkerMessage(
                message_type=WorkerMessageType.INPUT_SIGNALS,
                body=InputSignals(
                    request_id=request_id,
                    inputs=edges,
                    request_info=self.worker_graphs_manager.get_fwd_info(request_id, partition_name),
                    partition_name=partition_name
                ),
            )
            self.communicator.send(worker_id, message)
        if outputs.completed_worker_graph_ids:
            fwd_info = self.worker_graphs_manager.get_fwd_info(request_id, partition_name)
            if partition_name is None:
                partition_name = getattr(fwd_info, 'partition_name', 'default')
            req_info = self.worker_graphs_manager.per_request_info.get(request_id)
            p_done = (
                req_info.per_partition_info[partition_name].stream_partition_done \
                    and not node_speculatively_scheduled
            ) if req_info else False

            # Collect stream consumption info
            stream_consumed = {}
            if req_info:
                for edge_name, sbuf in req_info.stream_buffers.items():
                    stream_consumed[edge_name] = sbuf._consumed

            message = ConductorMessage(
                message_type=ConductorMessageType.WORKER_GRAPHS_DONE,
                body=WorkerGraphsDone(
                    request_id=request_id,
                    worker_graph_ids=outputs.completed_worker_graph_ids,
                    is_first_tp_rank=outputs.is_first_tp_rank,
                    persist_signals=self.worker_graphs_manager.flush_persist_signals(request_id),
                    new_tokens=self.worker_graphs_manager.flush_new_tokens(request_id),
                    output_signal_names=self.worker_graphs_manager.flush_output_signals(request_id),
                    per_label_seq_info=self.worker_graphs_manager.get_seq_info(request_id, partition_name),
                    partition_name=partition_name,
                    partition_done=p_done,
                    stream_tokens_consumed=stream_consumed,
                    output_loop_indices=self.worker_graphs_manager.get_output_loop_indices(request_id),
                    graph_timings=self.profile_info.per_rid_graph_timings.get(request_id, {}),
                    rx_info=self.tensor_manager.get_rx_info(request_id),
                    tx_info=self.tensor_manager.get_tx_info(request_id),
                ),
            )
            self.communicator.send("conductor", message)

    # ------------------------------------------------------------------
    # Main loop — async scheduling
    #
    # Pipeline shape:
    #   iter K (main thread):                          GPU thread
    #     CPU preamble  ───────────────► overlaps with execute_batch(N)
    #     speculate + build N+1
    #     await GPU(N).future Python return
    #     thread N's outputs → N+1's loop-back inputs
    #     submit GPU(N+1) ───────────────► execute_batch(N+1)
    #     _postprocess_batch(N) ─────────► overlap with GPU(N+1)
    #
    # Speculation scope (currently): AR engine only, intra-worker, 1-deep,
    # for rids whose loop is still continuing.
    # ------------------------------------------------------------------

    def _pre_plan_for_speculative_batch(
        self,
        engine,
        spec_node_batch: NodeBatch,
        prev_advance_event: "threading.Event | None",
    ) -> bool:
        """Dispatch entry point on the plan_executor for the speculative batch.

        Waits on ``prev_advance_event`` — set by the GPU thread RIGHT AFTER
        ``advance_seq_lens(prev)`` runs (~tens of µs into prev replay) —
        rather than on the full prev_future. This is the key to overlap:
        plan(N+1) starts as soon as alloc_manager state is post-(N), which
        is well before replay(N)'s GPU work finishes. plan(N+1) runs
        concurrent with the rest of replay(N)'s GPU kernels on the disjoint
        slot's wrapper buffers. await_plan on the GPU thread should drop
        to ~0 because plan(N+1) has finished long before replay(N+1)
        begins.

        Returns True if pre-planning was applied; False otherwise — the
        caller submits the spec batch with plan_future regardless, so a
        False return means the GPU thread will plan inline (no skip).
        """
        try:
            if prev_advance_event is not None:
                # Safety timeout — should fire well within 100ms in normal
                # operation. If it doesn't (e.g., GPU thread crashed),
                # bail out rather than block plan_executor forever.
                if not prev_advance_event.wait(timeout=10.0):
                    logger.warning(
                        "Worker %s: plan_executor timed out waiting for "
                        "prev advance_event; skipping pre-plan",
                        self.worker_id,
                    )
                    self._reset_skip_plan_flags(spec_node_batch)
                    return False
            return engine.pre_plan_for_batch(
                spec_node_batch,
                prev_completion_event=None,
            )
        except Exception:
            logger.exception("Worker %s: plan_executor pre-plan failed", self.worker_id)
            self._reset_skip_plan_flags(spec_node_batch)
            return False

    def _reset_skip_plan_flags(self, spec_node_batch: NodeBatch) -> None:
        """Clear pre-plan state on the SPECIFIC slot that
        ``pre_plan_for_batch`` targeted for ``spec_node_batch``.

        Used to recover from speculation drops / failures where the pre-
        plan was dispatched but the spec batch never reached the GPU
        thread — leaving entries in the slot's ``_pre_planned_labels``
        would cause the next real plan_attention call on that slot to
        short-circuit incorrectly.

        Slot-targeted (not worker-global) so that any other slot's
        valid in-flight pre-plan whose flags have not yet been consumed
        by the matching replay isn't stomped. The engine's
        ``reset_pre_plan_for_batch`` looks up the same (key, slot) the
        pre-plan path used; engines without a pre-plan surface inherit
        ``BaseEngine``'s no-op default.
        """
        engine = self.engine_manager.get_engine(spec_node_batch.node_name)
        engine.reset_pre_plan_for_batch(spec_node_batch)

    def _execute_on_gpu_thread(
        self,
        batch: ScheduledBatch,
        node_batch: NodeBatch,
        plan_future: Future | None = None,
        advance_event: "threading.Event | None" = None,
        stream: "torch.cuda.Stream | None" = None,
    ) -> NodeOutput:
        """Run the engine on a GPU executor thread.

        The NVTX range bracketing this call is ``synchronize=False`` —
        adding a ``cudaDeviceSynchronize`` at the marker boundary would
        drain the GPU on every iter and hide the overlap between
        post-processing and the next step's kernel execution.

        After ``execute_with_max_batch_size`` returns we record a CUDA event
        and stash it on the output. Downstream sync points on the main thread
        wait on this event.

        ``stream``: when None (the decode / normal path), execute and record
        the completion event on the DEFAULT stream — unchanged behavior. When
        a side stream is passed (MSTAR_SIDE_PREFILL), execute the batch and
        record the completion event on THAT stream, so the batch overlaps
        decode replays on the default stream and downstream token-
        materialization waits gate on the correct stream. The captured graphs
        carry no stream affinity (torch.cuda.graph captures on its own
        internal stream), so replay on a side stream is valid; prefill and
        decode use disjoint static I/O buffers and FlashInfer workspaces, so
        concurrent execution does not corrupt.
        """
        from mstar.utils.profiler import range_pop, range_push

        engine = self.engine_manager.get_engine(batch.node_name)
        logger.debug(
            "Executing batch for node %s on engine %s",
            node_batch.node_name, str(type(engine))
        )
        if self.enable_nvtx:
            range_push("worker.gpu_thread_start", synchronize=False)
            range_pop(synchronize=False)
        # Wait for the plan_executor's pre-planned wrapper.plan() call to
        # finish before running this batch — its results land on the captured
        # graph's persistent wrappers, and the next plan_attention call(s)
        # will see the matching label in _pre_planned_labels only because
        # plan_executor populated it. Wait releases the GIL.
        if plan_future is not None:
            if self.enable_nvtx:
                range_push("worker.gpu_thread.await_plan", synchronize=False)
            try:
                plan_future.result()
            finally:
                if self.enable_nvtx:
                    range_pop(synchronize=False)
        if self.enable_nvtx:
            range_push(
                f"worker[{self.worker_id}].node[{batch.node_name}].graph_walk[{batch.graph_walk}]",
                synchronize=False,
            )

        try:
            if stream is not None:
                # Side-stream execution (MSTAR_SIDE_PREFILL): run the whole
                # batch on the side stream so it overlaps default-stream decode
                # replays, and record the completion event on the SAME stream.
                with torch.cuda.stream(stream):
                    output = engine.execute_with_max_batch_size(node_batch)
                    if torch.cuda.is_available():
                        event = torch.cuda.Event()
                        event.record(stream)
                        output.completion_event = event
            else:
                output = engine.execute_with_max_batch_size(node_batch)
                if torch.cuda.is_available():
                    event = torch.cuda.Event()
                    event.record(torch.cuda.default_stream(self.device))
                    output.completion_event = event
            # MSTAR_MULTISTEP_DECODE: the single completion event, recorded
            # after the LAST step's replay was submitted, covers every step's
            # GPU work (all steps ran back-to-back on the default stream in one
            # submission). Share it with the extra-step NodeOutputs so their
            # postprocess sync waits on the same event instead of falling back
            # to a full default-stream synchronize. No-op when single-step.
            if output.extra_step_outputs and output.completion_event is not None:
                for extra in output.extra_step_outputs:
                    extra.completion_event = output.completion_event
            return output
        finally:
            # Safety net: ensure advance_event fires even if the engine
            # raised before reaching ``advance_seq_lens`` inside
            # ``_run_basic_batched``. Without this, a plan_executor waiting
            # on prev_advance_event would block forever on the failure path.
            if advance_event is not None:
                advance_event.set()
            # Same idea for launch_started_event: if the engine raised
            # before reaching the deep set site, release the main-thread
            # waiter early instead of making it eat the full timeout.
            launch_started_event = node_batch.metadata.get("launch_started_event")
            if launch_started_event is not None:
                launch_started_event.set()
            # Mirror engine-internal state (e.g. KV-cache seq_info) back
            # onto node_batch.per_request_info so the next iter's prep /
            # routing sees the updated values. Runs regardless of success,
            # allocation_failed, or an uncaught raise — finalize_batch
            # reads whatever state the engine actually reached.
            engine.finalize_batch(node_batch)
            if self.enable_nvtx:
                range_pop(synchronize=False)

    def _handle_allocation_failure(
        self, batch: ScheduledBatch, node_batch: NodeBatch
    ) -> None:
        """Push back nodes and hold the rids for backoff after KV OOM.

        Under TP, this runs on every rank of the TP group independently:
        admission decisions (``add_request`` / ``alloc`` / ``free``) are
        all driven by rank 0's scheduler and replicated via the
        ``ScheduleTPNode`` ZMQ broadcast, so the page allocator state is
        symmetric across ranks. Both rank 0 and followers raise
        ``AllocationFailedError`` on the same batch and both reach this
        function with the same ``batch_ids``; their local actions
        (push-back, hold) produce identical follower state.

        ``KVCacheEngine._verify_tp_kv_symmetry`` fails fast at startup if
        that invariant ever breaks.

        v2 caveat: this function does not yet coordinate ``_last_active``
        / eviction-victim selection across TP ranks. Wall-clock LRU can
        pick different victims per rank under contention, leading to
        request-id ↔ page-index drift and (eventually) asymmetric OOM on
        future reloads. Today's TP configs don't enable CPU offload, so
        the path isn't exercised; revisit when we light up offload + TP.
        """
        batch_ids = set(batch.node_objects.keys())
        victim_id = self._try_offload_cold_request(node_batch.node_name, batch_ids)

        # Push all batch nodes back to their queues
        for request_id, node in batch.node_objects.items():
            wg_id = batch.request_to_worker_graph[request_id]
            self.worker_graphs_manager.queues[wg_id].push_back_node(
                request_id, node
            )

        if victim_id is not None:
            self.scheduler.hold_requests([victim_id])
            logger.warning(
                "OOM on node=%s walk=%s: offloaded victim=%s, "
                "retrying %d remaining requests",
                batch.node_name, batch.graph_walk, victim_id,
                len(batch_ids) - (1 if victim_id in batch_ids else 0),
            )
        else:
            self.scheduler.hold_requests(list(batch_ids))
            logger.warning(
                "OOM on node=%s walk=%s: no offload possible, "
                "holding %d requests",
                batch.node_name, batch.graph_walk, len(batch_ids),
            )

    # ------------------------------------------------------------------
    # Multistep decode (MSTAR_MULTISTEP_DECODE)
    # ------------------------------------------------------------------

    def _eligible_multistep_decode(
        self, batch: ScheduledBatch, node_batch: NodeBatch
    ) -> int:
        """Return the number of decode steps to run in one submission for
        ``batch`` (>=2 when eligible, 1 otherwise).

        Eligibility (all must hold):
          * multistep enabled (flag > 1) and DIRECT_FEED on (checked at init);
          * uniform ``thinker_decode`` batch (single AR decode node/walk) — the
            same batch shape the BASIC_BATCHED decode graph captures and the
            only walk ``prepare_inputs_batched`` builds inline inputs for;
          * async scheduling allowed + not a TP node (same base gate as
            speculation — TP/eager can't do the inline replay chain);
          * every rid has a comfortable margin of loop iterations remaining.
            ``curr_iter`` counts only POSTPROCESSED iterations; when this runs
            there can be another in-flight batch (the current ``pending``, whose
            postprocess happens after this submit) plus THIS batch, each up to
            ``num_steps`` steps. Requiring ``max_iters - curr_iter >=
            2 * num_steps`` guarantees no rid overshoots ``max_iters`` even with
            one other multistep batch in flight — keeping the max-tokens
            overshoot within the accepted ``num_steps - 1`` (constraint: EOS /
            last-token late by up to num_steps-1, discarded cleanly by the
            stopped-rid guards). A rid missing loop state is treated as
            ineligible (fall back to single-step for the whole batch — the
            captured graph runs one shape for all rids).

        The in-graph seen-token penalty also forces single-step; that gate
        lives in the runner (it owns the captured-config flag), which reduces
        num_steps to 1 defensively even if this returns 2. The postprocess
        loop keys off the steps ACTUALLY returned, so a runner downgrade is
        handled transparently.
        """
        num_steps = self._multistep_decode
        if num_steps <= 1:
            return 1
        if batch.graph_walk != "thinker_decode":
            return 1
        if not self._can_speculate(batch):
            return 1
        for rid in node_batch.request_ids:
            wgio = self._get_wgio_for_rid(batch, rid)
            # Find the loop governing this decode node for this rid. All rids
            # in a uniform decode batch share the same loop; bail to
            # single-step if any rid's loop can't be resolved or is too close
            # to its max_iters.
            if batch.node_objects.get(rid) is None:
                return 1
            rid_loop = None
            for loop in wgio.loops.values():
                if batch.node_name in loop.get_nodes():
                    rid_loop = loop
                    break
            if rid_loop is None:
                return 1
            if rid_loop._finish_signal or rid_loop.is_done:
                return 1
            # curr_iter counts postprocessed iters only. Reserve margin for one
            # other in-flight multistep batch (pending) + this batch, each up to
            # num_steps steps, so no rid runs past max_iters by more than
            # num_steps-1 (see docstring).
            remaining = rid_loop.max_iters - rid_loop.curr_iter
            if remaining < 2 * num_steps:
                return 1
            # Also honor the per-request ``max_tokens`` stop that check_stop
            # enforces on ``dynamic_loop_iter_counts`` — it can be tighter than
            # the loop's ``max_iters``. Same 2*num_steps margin so a rid never
            # overshoots max_tokens by more than the accepted num_steps-1.
            info = node_batch.per_request_info.get(rid)
            if info is not None:
                done_iters = info.dynamic_loop_iter_counts.get(
                    "thinker_decode_loop", 0
                )
                if info.max_tokens - done_iters < 2 * num_steps:
                    return 1
        return num_steps

    # ------------------------------------------------------------------
    # Speculation
    # ------------------------------------------------------------------

    def _can_speculate(self, batch: ScheduledBatch) -> bool:
        if any(
            not node.enable_async_scheduling for node in batch.node_objects.values()
        ) or batch.node_name in self.tp_nodes:
            # disable speculation for TP nodes for now
            return False
        return True

    def _is_side_eligible(self, batch: ScheduledBatch) -> bool:
        """Whether ``batch`` may run on the side stream concurrently with the
        decode chain (MSTAR_SIDE_PREFILL).

        Eligible batches are the ones whose GPU work we want to hide behind
        decode: thinker-PREFILL walks (KV_CACHE engine, graph_walk starting
        with ``prefill``) and STATELESS encoder nodes (which route into a
        prefill; they may or may not be co-located depending on the 2-GPU
        config variant). Decode (``thinker_decode``) is never side-eligible —
        it is the chain we overlap AGAINST, not a batch to overlap.

        TP nodes are excluded: TP scheduling is driven by rank-0 broadcast
        (ScheduleTPNode) and the follower ranks execute on their own GPU
        threads with default-stream ordering assumptions; running a TP batch on
        a side stream on rank 0 only would desynchronize the group. Keep the
        side path single-rank (non-TP) prefill/encoder work.
        """
        if not self._side_prefill:
            return False
        if not batch.node_objects:
            return False
        if batch.node_name in self.tp_nodes:
            return False
        engine = self.engine_manager.get_engine(batch.node_name)
        etype = engine.engine_type()
        if etype == EngineType.STATELESS:
            return True
        if etype == EngineType.KV_CACHE and batch.graph_walk.startswith("prefill"):
            return True
        return False

    def _reap_side_if_done(self, pending_side: "PendingSide | None") -> "PendingSide | None":
        """If a side batch has finished on the side stream, postprocess it on
        the MAIN thread and return None; otherwise return it unchanged.

        Opportunistic: never blocks on the future. Postprocess reuses the
        normal routing path (_postprocess_batch), whose completion_event
        handling drains the side stream before routing — see the correctness
        note in __init__."""
        if pending_side is None:
            return None
        if not pending_side.future.done():
            return pending_side
        try:
            output: NodeOutput = pending_side.future.result()
            for node in pending_side.batch.node_objects.values():
                node._speculatively_scheduled = False
            if output.allocation_failed:
                # KV OOM on the side prefill: rehabilitate exactly like the
                # decode path (push nodes back + hold rids). Does not touch the
                # decode chain — the failed rids are disjoint from live decode.
                self._handle_allocation_failure(
                    pending_side.batch, pending_side.node_batch
                )
            else:
                # _postprocess_batch unconditionally clears
                # self._pending_loop_stops at its tail (they are a one-iter
                # decode-speculation mechanism, consumed only by the NEXT
                # decode speculative_new_iter postprocess). This side reap runs
                # at the TOP of the loop, BEFORE the current iter's decode
                # speculation consumes the stops the previous decode iter set —
                # so letting the side postprocess clear them would drop a
                # legitimate decode loop-stop. Prefill batches never set
                # speculative_new_iter and any prefill-walk stops they add are
                # never consumed, so the decode chain's pending stops must pass
                # through the side postprocess untouched: snapshot and restore.
                saved_loop_stops = set(self._pending_loop_stops)
                self._postprocess_batch(
                    PendingBatch(
                        batch=pending_side.batch,
                        node_batch=pending_side.node_batch,
                        node_name=pending_side.node_name,
                        partition=pending_side.partition,
                        graph_walk=pending_side.graph_walk,
                        future=pending_side.future,
                    ),
                    output,
                )
                self._pending_loop_stops = saved_loop_stops
        except Exception:
            logger.exception(
                "Worker %s: side prefill batch failed", self.worker_id
            )
        finally:
            self._side_in_flight_rids = set()
        return None

    def _maybe_dispatch_side(
        self,
        pending_side: "PendingSide | None",
        side_executor: "ThreadPoolExecutor | None",
        exclude_target: "tuple[str, str] | None",
    ) -> "PendingSide | None":
        """When a decode chain is active and no side batch is in flight, try to
        schedule a prefill/encoder batch and run it on the side stream.

        ``exclude_target`` is the active decode (node, walk) so the scheduler
        never hands back the decode group. Because get_next_batch POPS the
        chosen nodes out of the ready queue, the subsequent speculation
        has_ready_excluding won't re-see them — so dispatching here does not
        disturb the decode speculation chain. Returns the new pending_side (or
        the unchanged one if nothing was dispatched).

        All scheduling stays on the main thread (this method is only called
        from run()); the side executor solely EXECUTES the built batch."""
        if not self._side_prefill or side_executor is None:
            return pending_side
        if pending_side is not None:
            return pending_side  # one side batch in flight at a time
        with self._scheduler_lock:
            batch = self.scheduler.get_next_batch(
                self.worker_graphs_manager,
                exclude_target=exclude_target,
            )
        if batch is None:
            return pending_side
        if not self._is_side_eligible(batch):
            # Not a prefill/encoder we want on the side stream. We already
            # popped it from the queue; push its nodes back so the normal
            # (main-stream) path schedules it next iter.
            self._pushback_scheduled_batch(batch)
            return pending_side

        node_batch = self._build_node_batch(batch)
        # Force this batch down the eager / batched path, NEVER the captured
        # decode CUDA graph. KVCacheEngine._can_use_cuda_graph returns False
        # when this flag is set: the captured graph shares interned static I/O
        # buffers across slots and mutates a single-writer next_slot counter,
        # both of which a concurrent side replay would race. The eager path
        # uses FlashInfer workspace label "main", disjoint from decode's
        # per-slot labels, so it is safe concurrent with decode. The flag rides
        # NodeBatch.metadata, which execute_with_max_batch_size propagates to
        # every minibatch, so the gate holds even under max-batch-size splits.
        node_batch.metadata["side_stream"] = True
        batch_partition = self.worker_graphs_manager.get_partition_for_node(
            batch.node_name
        )
        for request_id, req_info in node_batch.per_request_info.items():
            req_info.dynamic_loop_iter_counts.update(
                self.worker_graphs_manager.get_dynamic_loop_iters(
                    request_id, partition=batch_partition,
                )
            )
        # In-flight bookkeeping: defer removes for these rids until the side
        # batch is postprocessed, and keep the nodes off the ready queue while
        # executing (same guard decode uses).
        for node in batch.node_objects.values():
            node._speculatively_scheduled = True
        self._side_in_flight_rids = set(batch.node_objects.keys())
        self.maybe_send_zmq_to_tp_followers(node_batch)
        future = side_executor.submit(
            self._execute_on_gpu_thread,
            batch, node_batch,
            None,            # plan_future — side prefill plans inline (eager)
            None,            # advance_event — not part of the spec chain
            self._side_stream,
        )
        self.wakeup_event.register_future(future)
        logger.debug(
            "Side-dispatch: %s %s", batch.node_name, node_batch.request_ids
        )
        return PendingSide(
            batch=batch,
            node_batch=node_batch,
            node_name=batch.node_name,
            partition=batch_partition,
            graph_walk=batch.graph_walk,
            future=future,
        )

    def _pushback_scheduled_batch(self, batch: ScheduledBatch) -> None:
        """Return a popped-but-not-run batch's nodes to their ready queues so a
        later schedule can pick them up. Used when a side-dispatch candidate
        turns out not to be side-eligible."""
        for rid, node in batch.node_objects.items():
            wg_id = batch.request_to_worker_graph.get(rid)
            if wg_id is None:
                continue
            queue = self.worker_graphs_manager.queues.get(wg_id)
            if queue is None:
                continue
            queue.push_back_node(rid, node)

    def _get_wgio_for_rid(self, batch: ScheduledBatch, rid: str):
        """Per-rid WorkerGraphIO for the wg that owns this rid in this batch.
        """
        wg_id = batch.request_to_worker_graph[rid]
        return self.worker_graphs_manager.queues[wg_id].per_request_queues[rid]

    def _get_input_tensors(
        self, rid: str, node: GraphNode, check_next_iter: bool
    ) -> NameToTensorList:
        inputs = node.ready_next_iter.ready_inputs if check_next_iter \
            else node.ready_signals.ready_inputs
        tensors = {}
        for input_name, edge in inputs.items():
            tensors[input_name] = [
                self.tensor_manager.get_tensor(
                    request_id=rid, uuid=info.uuid,
                )
                for info in edge.tensor_info
            ]
        return tensors

    def _try_speculate_next(
        self,
        pending: PendingBatch
    ) -> Speculation | None:
        """Build a speculative N+1 batch + node_batch, by checking which nodes
        will become ready after the current batch's outputs are ingested.

        The speculated batch is a merge of:
          * **continuing** rids (subset of batch_N still alive, not
            pending-stop / pending-remove) — placeholder inputs are gathered
            from the registry now (``_get_input_tensors``) and the entries
            tied to ``consumed_edges`` are overwritten with batch_N's outputs
            after await by ``_thread_outputs_to_speculative``.
          * **fresh** rids — newly-arrived requests whose spec-target node
            is ready in the queue right now. Their inputs come from the
            usual tensor_manager path (same as ``_build_node_batch``).
            Without this merge, new rids have to wait for the entire
            current speculation chain to drain before they can be scheduled.
        """
        batch_N = pending.batch
        partition_N = pending.partition
        graph_walk = pending.graph_walk

        # sample node and RID to see which node we will be speculating
        # (TODO: refine this to be, e.g., a majority vote)
        rid, sample_node = next(iter(batch_N.node_objects.items()))
        wgio = self._get_wgio_for_rid(batch_N, rid)

        # If sample_node has no outputs at all, it can't feed any spec target.
        if not sample_node.outputs:
            return
        ready_for_spec = wgio.ingest_for_speculation(
            sample_node.outputs, sample_node.name
        )
        wgio.clear_speculative_inputs()

        # Filter out destinations that aren't speculation candidates.
        #
        # * ``info.node_name in self.tp_nodes`` — TP nodes don't support
        #   speculation in v1 (rank-0 schedules; followers can't initiate).
        # * ``not wgio.nodes[info.node_name].enable_async_scheduling`` — the
        #   destination node opts out of async scheduling. Mirrors the
        #   source-side check in ``_can_speculate``; without this, a
        #   destination that's structurally ineligible (e.g. a node whose
        #   downstream graph isn't speculation-safe) could still be picked,
        #   then dropped per-rid further down.
        ready_for_spec = [
            info for info in ready_for_spec
            if info.node_name not in self.tp_nodes
            and wgio.nodes[info.node_name].enable_async_scheduling
        ]

        if not ready_for_spec:
            return # no nodes can be speculated

        # TODO: use the microscheduler to break ties when ready_for_spec
        # contains multiple ready nodes
        spec_node_info = ready_for_spec[0]
        speculating_same_node = spec_node_info.node_name == batch_N.node_name

        continuing = []
        new_node_objects: dict[str, GraphNode] = {}
        new_request_to_worker_graph: dict[str, str] = {}
        per_request_inputs: dict[str, NameToTensorList] = {}
        consumed_streaming_edges: dict[str, GraphEdge] = {}
        for rid, batch_N_node in batch_N.node_objects.items():
            wgio = self._get_wgio_for_rid(batch_N, rid)
            loop = wgio.loops.get(spec_node_info.loop_name)

            # check conditions where the rid cannot be furtuer speculated
            already_removed = rid in self._pending_removes
            already_stopped = spec_node_info.is_new_loop_iter and PendingLoopStop(
                rid, graph_walk, spec_node_info.loop_name
            ) in self._pending_loop_stops
            is_stopping = spec_node_info.is_new_loop_iter and loop is not None and (
                loop.curr_iter + 1 >= loop.max_iters or loop._finish_signal
            )
            if already_removed or already_stopped or is_stopping:
                # Loop/request has already finished, don't speculate further work
                continue

            # If the speculation is contingent on streaming edges, ingest the
            # appropriate streaming edges
            node = wgio.nodes[spec_node_info.node_name]

            # temporarily set to prevent ingesting streaming inputs from re-adding the node to
            # the ready queue
            node._speculatively_scheduled = True
            streaming_edges = self._poll_stream_buffers_for_speculation(
                rid, spec_node_info.node_name
            )
            # Track which slot each ingest landed in so we can roll back if
            # the readiness check below fails. ``ingest_input`` returns success
            # without telling us which slot it used, so peek the slot state
            # before the call.
            ingested_into_ready_signals: list[GraphEdge] = []
            ingested_into_ready_next_iter: list[GraphEdge] = []
            for edge in streaming_edges:
                already_in_ready_signals = (
                    edge.name in node.ready_signals.ready_names
                )
                if node.ingest_input(
                    edge, can_buffer=speculating_same_node
                ):
                    if already_in_ready_signals:
                        ingested_into_ready_next_iter.append(edge)
                    else:
                        ingested_into_ready_signals.append(edge)
                else:
                    self._return_speculative_streaming_edge(rid, edge)

            # Check if the node is ready after ingesting the streaming edges
            wgio.ingest_for_speculation(
                batch_N_node.outputs, batch_N_node.name
            )
            fully_ready = node.is_ready_for_speculation(
                check_next_iter=speculating_same_node,
                allow_streaming=False
            )
            wgio.clear_speculative_inputs()
            node._speculatively_scheduled = False # reset in case this rid gets dropped
            if not fully_ready:
                # Roll back the streaming edges we just ingested so the
                # chunks don't sit in the spec target's ready_signals /
                # ready_next_iter unused. Return each chunk to its
                # StreamBuffer's uningested cache so a future scheduling
                # of this node consumes it normally. The registry's
                # ``ready_names`` / ``ready_next_iter`` sets weren't
                # touched (gated by ``_speculatively_scheduled=True``
                # above), so no registry-state rollback is needed.
                for edge in ingested_into_ready_signals:
                    node.ready_signals.remove(edge.name)
                    self._return_speculative_streaming_edge(rid, edge)
                for edge in ingested_into_ready_next_iter:
                    node.ready_next_iter.remove(edge.name)
                    self._return_speculative_streaming_edge(rid, edge)
                continue

            # prepare speculative batch
            continuing.append(rid)
            new_node_objects[rid] = node
            new_request_to_worker_graph[rid] = wgio.wg_id
            per_request_inputs[rid] = self._get_input_tensors(
                rid, node, check_next_iter=speculating_same_node,
            )
            consumed_streaming_edges[rid] = ingested_into_ready_next_iter + ingested_into_ready_signals

        if not continuing:
            return None

        # Edges that the spec batch effectively "consumed" from batch_N's
        # outputs: every output of sample_node whose destination is the spec node
        consumed_edges: set[tuple[str, str]] = {
            (edge.name, edge.next_node)
            for edge in sample_node.outputs
            if edge.next_node == spec_node_info.node_name
        }

        # Merge in fresh rids whose spec-target node is ready right now
        # Speculation only consumes work compatible with the spec target. In
        # partitioned models, unrelated ready work stays queued for
        # the normal scheduler path.
        fresh_batch = self.scheduler.get_next_batch(
            self.worker_graphs_manager,
            target_node_name=spec_node_info.node_name,
            target_graph_walk=batch_N.graph_walk,
        )

        if fresh_batch is not None:
            for rid, node in fresh_batch.node_objects.items():
                if rid in new_node_objects:
                    # Shouldn't happen — continuing rids are held by the
                    # in-flight step and shouldn't be in ready queues —
                    # but if it does, the in-flight rid wins.
                    wg_id = fresh_batch.request_to_worker_graph[rid]
                    self.worker_graphs_manager.queues[wg_id].push_back_node(rid, node)
                    continue

                per_request_inputs[rid] = self._get_input_tensors(
                    rid, node, check_next_iter=False
                )
                new_node_objects[rid] = node
                new_request_to_worker_graph[rid] = (
                    fresh_batch.request_to_worker_graph[rid]
                )

        spec_batch = ScheduledBatch(
            node_name=spec_node_info.node_name,
            graph_walk=batch_N.graph_walk,
            node_objects=new_node_objects,
            request_to_worker_graph=new_request_to_worker_graph,
        )

        request_ids = list(new_node_objects.keys())
        spec_node_batch = NodeBatch(
            node_name=spec_node_info.node_name,
            graph_walk=batch_N.graph_walk,
            request_ids=request_ids,
            per_request_input_tensors=per_request_inputs,
            per_request_info={
                rid: self.worker_graphs_manager.get_fwd_info(
                    rid, partition_N
                ) for rid in request_ids
            },
            final_stream_rids={
                rid for rid, edges in consumed_streaming_edges.items()
                if any(e._final_stream_chunk for e in edges)
            },
        )

        logger.debug(f"Speculating: {spec_node_info.node_name} {spec_node_batch.request_ids}")

        return Speculation(
            scheduled_batch=spec_batch,
            node_batch=spec_node_batch,
            consumed_edges=consumed_edges,
            continuing_rids=set(continuing),
            partition=pending.partition,
            is_new_iter=spec_node_info.is_new_loop_iter,
            is_same_node=speculating_same_node,
            loop_name=spec_node_info.loop_name,
            consumed_streaming_edges=consumed_streaming_edges
        )

    def _thread_outputs_to_speculative(
        self, speculation: Speculation, output_N: NodeOutput
    ):
        threaded_continuing: set[str] = set()
        dropped: set[str] = set()

        # MSTAR_DIRECT_FEED fast path. Precondition for using the batched
        # tensor at all: the flag is on, this is a same-node loop-back
        # (uniform thinker_decode) step, and batch_N's engine exposed the
        # [bs] sampled-tokens tensor + its rid order. Any consumed edge NOT
        # covered by that tensor (a rid missing from batched_sampled_rids, or
        # a consumed edge that isn't the loop-back token) falls through to the
        # per-rid output-map copy below — so a partial/absent tensor never
        # drops a rid, it just uses the slower (but identical-valued) route.
        # The batched tensor's rows are the SAME clone the per-rid new_token /
        # text_inputs views point at (cuda_graph_runner._sample_and_remap), so
        # sampled[i:i+1] is byte-identical to rid_outputs["text_inputs"][0];
        # it's a fresh per-step clone (no aliasing with FlashInfer's reused
        # sampling buffer — see the clone rationale there), so the row views
        # stay valid until the spec batch consumes them.
        row_for_rid: dict[str, "torch.Tensor"] = {}
        direct_feed_active = (
            self._direct_feed
            and speculation.is_same_node
            and output_N.batched_sampled_tokens is not None
            and output_N.batched_sampled_rids is not None
        )
        if direct_feed_active:
            sampled = output_N.batched_sampled_tokens
            for i, r in enumerate(output_N.batched_sampled_rids):
                row_for_rid[r] = sampled[i:i + 1]

        for rid in list(speculation.node_batch.request_ids):
            if rid not in speculation.continuing_rids:
                continue  # fresh rid — inputs already gathered.
            rid_outputs = output_N.per_request_output_tensors.get(rid, {})
            row_view = row_for_rid.get(rid) if direct_feed_active else None
            ok = True
            for input_name, _ in speculation.consumed_edges:
                tensors = rid_outputs.get(input_name, [])
                # Substitute the batched row ONLY when it is provably the same
                # value as this edge's per-rid output: a single 1-element token
                # view (the loop-back sampled token). Any other loop-back edge
                # (multi-tensor / non-token) takes the per-rid copy path so a
                # future same-node loop with a non-token loop-back stays correct.
                if (
                    row_view is not None
                    and len(tensors) == 1
                    and torch.is_tensor(tensors[0])
                    and tensors[0].numel() == 1
                ):
                    speculation.node_batch.per_request_input_tensors[rid][input_name] \
                        = [row_view]
                    continue
                if not tensors:
                    ok = False
                    break
                speculation.node_batch.per_request_input_tensors[rid][input_name] \
                    = list(tensors)
            if ok:
                threaded_continuing.add(rid)
            else:
                dropped.add(rid)

        if dropped:
            logger.warning(
                "Speculation: dropped rids %s (no loop-back output from N)",
                sorted(dropped),
            )
            speculation.node_batch.request_ids = [
                r for r in speculation.node_batch.request_ids if r not in dropped
            ]
            for r in dropped:
                speculation.node_batch.per_request_input_tensors.pop(r, None)
                speculation.node_batch.per_request_info.pop(r, None)
                speculation.scheduled_batch.request_to_worker_graph.pop(r, None)
                speculation.scheduled_batch.node_objects.pop(r, None)
                for edge in speculation.consumed_streaming_edges.get(rid, []):
                    self._return_speculative_streaming_edge(rid, edge)
                speculation.consumed_streaming_edges.pop(rid, None)
        speculation.continuing_rids = threaded_continuing
        speculation.dropped = dropped

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------
    def _cleanup_consumed_inputs(self, batch: ScheduledBatch) -> None:
        """Free input tensors that were consumed by the just-executed node."""
        for node in batch.node_objects.values():
            node.ready_signals.clear()


    def _prematerialized_new_tokens(
        self, cpu_output, rid: str,
    ) -> dict[str, list[int]] | None:
        """Extract prematerialized new-token ints for ``rid`` from check_stop's
        CPU output.

        Reuses check_stop's side-stream D→H copies for the new-token ints.
        Without this, buffer_new_tokens does a per-rid ``get_tensor().cpu()``
        — a default-stream sync per request per step (32/step at B32) that
        also serializes against the in-flight speculative step's kernels on
        the default stream. Only integer, non-CUDA tensors qualify; audio /
        multimodal (float / large) outputs are never included.
        """
        rid_cpu = cpu_output.per_request_output_tensors.get(rid)
        if not isinstance(rid_cpu, dict):
            return None
        prem: dict[str, list[int]] = {}
        for name, tensors in rid_cpu.items():
            if (
                isinstance(tensors, list)
                and tensors
                and all(
                    torch.is_tensor(t)
                    and not t.is_cuda
                    and not t.is_floating_point()
                    for t in tensors
                )
            ):
                prem[name] = [
                    int(v) for t in tensors for v in t.flatten().tolist()
                ]
        return prem

    def _postprocess_batch(
        self, batch_N: PendingBatch,
        output: NodeOutput,
    ):
        if self.enable_nvtx:
            range_push("worker.postprocess.cleanup_inputs", synchronize=False)
        self._cleanup_consumed_inputs(batch_N.batch)
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.pending_loop_stops", synchronize=False)
        # If any nodes in the batch have "overstayed" their loop stop, then make
        # sure to not route their outputs
        valid_rids = set(batch_N.node_batch.request_ids)
        if batch_N.speculative_new_iter:
            for pending_stop in self._pending_loop_stops:
                if pending_stop.loop_name != batch_N.loop_name \
                        or pending_stop.graph_walk != batch_N.graph_walk \
                        or pending_stop.rid not in batch_N.node_batch.request_ids:
                    continue
                stopped_rid = pending_stop.rid
                if stopped_rid not in batch_N.batch.node_objects:
                    continue
                output.per_request_output_tensors.pop(stopped_rid, None)
                valid_rids.discard(stopped_rid)
                batch_N.batch.node_objects.pop(stopped_rid)
                batch_N.batch.request_to_worker_graph.pop(stopped_rid)
                batch_N.node_batch.per_request_info.pop(stopped_rid)
        batch_N.node_batch.request_ids = list(valid_rids)

        # pending stops are only needed for one iteration, so can be cleared now
        self._pending_loop_stops.clear()

        per_req_nested_idxs = {
            rid: self.worker_graphs_manager.get_nested_loop_idxs_for_node(
                rid, batch_N.partition, batch_N.node_name
            ) for rid in batch_N.node_batch.request_ids
        }

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.update_lru", synchronize=False)

        # Update LRU
        t = _time.monotonic()
        for rid in batch_N.node_batch.request_ids:
            self._last_active[(rid, batch_N.node_name)] = t

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.synchronize_completion_event", synchronize=False)

        # Wait for batch N's completion event before proceeding
        # TODO: may need to refine this based on how it affects performance?
        if torch.cuda.is_available() and batch_N.batch.node_objects:
            if output.completion_event is not None:
                if self.enable_nvtx:
                    range_push("worker.postprocess.completion_event_sync", synchronize=False)
                output.completion_event.synchronize()
                if self.enable_nvtx:
                    range_pop(synchronize=False)
            else:
                torch.cuda.default_stream().synchronize()

        if self.enable_prof:
            batch_N.node_batch.exec_timings.fwd_end = time.perf_counter()

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.check_stop", synchronize=False)

        for rid, req_info in batch_N.node_batch.per_request_info.items():
            new_iters = self.worker_graphs_manager.get_dynamic_loop_iters(
                rid, partition=batch_N.partition,
            )
            req_info.dynamic_loop_iter_counts.update(new_iters)

        # Check for stops
        engine = self.engine_manager.get_engine(batch_N.node_name)
        cpu_output = self._prematerialize_for_check_stop(output)
        new_stops = engine.check_stop_for_batch(batch_N.node_batch, cpu_output)

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.stop_loops", synchronize=False)

        # Stop loops, if applicable
        for rid, loop_names in new_stops.items():
            self.worker_graphs_manager.stop_loops(
                rid, partition=batch_N.partition,
                loop_names=loop_names,
                req_info=batch_N.node_batch.per_request_info[rid],
                last_node_run=batch_N.node_name
            )
            self._pending_loop_stops.update([
                PendingLoopStop(
                    rid=rid,
                    graph_walk=batch_N.graph_walk,
                    loop_name=name
                ) for name in loop_names
            ])

            # Send "loop done" messages to peer workers (small ZMQ msgs)
            stop_loop_workers: dict[str, set[str]] = {}
            for loop_name in loop_names:
                for worker in self.worker_graphs_manager.get_dyn_loop_workers(
                    rid, batch_N.partition, loop_name
                ):
                    stop_loop_workers.setdefault(worker, set()).add(loop_name)
            for worker, loop_names in stop_loop_workers.items():
                if worker == self.worker_id:
                    continue
                self.communicator.send(
                    entity_id=worker,
                    msg=WorkerMessage(
                        message_type=WorkerMessageType.STOP_LOOPS,
                        body=StopLoops(
                            request_id=rid,
                            loop_names=loop_names,
                            loop_stop_times=batch_N.node_batch.per_request_info[rid].loop_stop_times,
                            partition_name=batch_N.partition
                        )
                    )
                )

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.route_outputs", synchronize=False)
        # Mark nodes complete and route
        routing_per_request: dict[str, NodeOutputRouting] = {}
        per_request_uuids: dict[str, set[str]] = {}
        for rid, wg_id in batch_N.batch.request_to_worker_graph.items():
            # Store output tensors before marking the node as complete so that
            # loop outputs can be buffered properly.
            req_output_tensors = output.per_request_output_tensors.get(rid)
            node = batch_N.batch.node_objects[rid]
            node.reset_outputs() # reset stale outputs
            if req_output_tensors:
                graph_node_info = self.tensor_manager.store_and_populate_graph_edges(
                    request_id=rid,
                    tensors=req_output_tensors,
                    graph_edges=node.outputs,
                    node_name=node.name,
                    graph_walk=batch_N.graph_walk,
                    skip_cuda_sync=True,
                    skip_ref_count=True,
                )
                per_request_uuids[rid] = {
                    info.uuid for infos in graph_node_info.values() for info in infos
                }

            completion_output = self.worker_graphs_manager.mark_node_complete(
                rid, wg_id, batch_N.node_name
            )
            real_outputs = [edge.clone() for edge in completion_output.output_edges]

            routing_per_request[rid] = self.worker_graphs_manager.process_node_outputs(
                rid, node_name=batch_N.node_name,
                outputs=real_outputs, graph_walk=batch_N.graph_walk
            )

            if rid in per_request_uuids:
                routing = routing_per_request[rid]
                routed_edges = (
                    routing.routed_to_this_worker_graph
                    + routing.persist
                    + routing.emit_to_client
                    + routing.streaming_local
                    + sum(routing.to_workers.values(), start=[])
                    + sum(routing.streaming_to_workers.values(), start=[])
                )
                self.tensor_manager.set_output_ref_counts(
                    rid, per_request_uuids[rid], routed_edges
                )

        # Build the prematerialized new-token ints once, before register_outputs,
        # so both the SHM-skip decision (register_outputs) and the inline send
        # (_send_outputs) see the same per-rid prem dict.
        prem_per_request: dict[str, dict[str, list[int]] | None] = {
            rid: self._prematerialized_new_tokens(cpu_output, rid)
            for rid in routing_per_request
        }

        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.register_outputs", synchronize=False)
        self._register_outputs(
            batch_N.batch, routing_per_request,
            prematerialized_per_request=prem_per_request,
        )

        # send outputs
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.send_outputs", synchronize=False)

        # The consuming pass (not the earlier ingest) reports the partition
        # done, so it rides this pass's WGD with the final output loop index.
        for rid in batch_N.node_batch.final_stream_rids:
            req_info = self.worker_graphs_manager.per_request_info.get(rid)
            if req_info is not None:
                req_info.per_partition_info[batch_N.partition].stream_partition_done = True

        # set this before send_outputs so that we can send updated profiling info to the conductor
        if self.enable_prof:
            self.profile_info.register_end(
                batch_N.node_batch.node_name,
                batch_N.node_batch.graph_walk,
                batch_N.node_batch.request_ids,
                batch_N.node_batch.exec_timings,
            )
        # MSTAR_BATCH_EMIT: collect every rid's qualifying inline emit results
        # into one list and send them as a single result_tensors_batch message
        # after the loop, instead of one result_tensors message per rid. When
        # off, batch_collector stays None and each rid sends its own messages
        # exactly as before (byte-identical).
        batch_collector: list[ResultTensors] | None = (
            [] if self._batch_emit else None
        )
        for rid, routing in routing_per_request.items():
            self._send_outputs(
                rid, routing,
                nested_loop_indices=per_req_nested_idxs[rid],
                graph_walk=batch_N.graph_walk,
                partition_name=batch_N.partition,
                prematerialized_new_tokens=prem_per_request[rid],
                node_speculatively_scheduled=batch_N.batch.node_objects[rid]._speculatively_scheduled,
                batch_collector=batch_collector,
            )
        if batch_collector:
            self.communicator.send(
                "api_server",
                APIServerMessage(
                    message_type="result_tensors_batch",
                    body=ResultTensorsBatch(items=batch_collector),
                ),
            )

        if self.enable_nvtx:
            range_pop(synchronize=False)

        return routing_per_request

    def _d2h_new_tokens(
        self,
        tensors: list[torch.Tensor],
        completion_event: "torch.cuda.Event | None",
    ) -> list[int]:
        """Batched D→H copy of new-token tensors, gated on GPU(N)'s
        completion event so it does not block on GPU(N+1) (which is queued
        on default stream behind GPU(N)).

        Falls back to the simple ``torch.cat([...]).cpu()`` when CUDA is
        unavailable, the tensors are already on CPU, or no completion event
        was recorded (non-CUDA execution).

        Safety: assumes ``tensors`` are fresh allocations (not views into
        CUDA-graph static buffers that GPU(N+1) will overwrite). Sampler
        outputs from FlashInfer's ``top_p_sampling_from_probs`` qualify;
        if a future change makes the new-token tensor a static-buffer
        view, this needs an extra clone-on-default-stream-before-event-
        record step on the GPU thread.
        """
        if not tensors:
            return []
        first = tensors[0]
        on_cuda = first.is_cuda and torch.cuda.is_available()
        if not on_cuda or completion_event is None:
            return torch.cat([t.flatten() for t in tensors]).cpu().tolist()

        if self._d2h_stream is None:
            self._d2h_stream = torch.cuda.Stream(device=self.device)
        side = self._d2h_stream
        side.wait_event(completion_event)
        with torch.cuda.stream(side):
            flat_gpu = torch.cat([t.flatten() for t in tensors])
            flat_cpu = self._get_pinned_d2h_buffer(
                "new_tokens", flat_gpu.shape, flat_gpu.dtype,
            )
            flat_cpu.copy_(flat_gpu, non_blocking=True)
        side.synchronize()
        return flat_cpu.tolist()

    def _get_pinned_d2h_buffer(
        self,
        purpose: str,
        shape: torch.Size | tuple[int, ...],
        dtype: torch.dtype,
        index: int = 0,
    ) -> torch.Tensor:
        key = (purpose, dtype, tuple(shape))
        buffers = self._pinned_d2h_buffers[key]
        while len(buffers) <= index:
            buffers.append(
                torch.empty(key[2], dtype=dtype, device="cpu", pin_memory=True)
            )
        return buffers[index]

    def _prematerialize_for_check_stop(
        self,
        output: NodeOutput,
    ) -> NodeOutput:
        """Side-stream D→H of every CUDA tensor in
        ``output.per_request_output_tensors`` so the subsequent
        ``check_stop`` reads (typically ``.item()`` on the sampled token)
        don't trigger a default-stream sync. With same-thread async,
        GPU(N+1)'s kernels are already queued on default stream behind
        N's outputs by the time we get here — a default-stream sync would
        block waiting for N+1 to finish, defeating the overlap.

        Returns a fresh ``NodeOutput`` with CPU tensors for the per-rid
        outputs, sharing the original's allocation_failed / event fields.
        Skipped (returns ``output`` unchanged) when there's no completion
        event (CPU execution) or when CUDA is unavailable.

        AR engines emit small per-rid output dicts (sampled token + maybe
        a code) so the cost is negligible. If a future engine emits large
        tensors here (e.g. activations), revisit.
        """
        if not torch.cuda.is_available() or output.completion_event is None:
            return output
        if not output.per_request_output_tensors:
            return output

        if self._d2h_stream is None:
            self._d2h_stream = torch.cuda.Stream(device=self.device)
        side = self._d2h_stream
        side.wait_event(output.completion_event)

        # Fast path: the common AR-decode shape is exactly one small
        # same-shaped tensor per rid under one key (new_token). Batch the
        # whole step into a single cat + one pinned D2H instead of a
        # per-rid copy loop (32 tiny copies/step at B32).
        per_rid = output.per_request_output_tensors
        rids = list(per_rid.keys())
        uniform_key: str | None = None
        if rids and all(
            isinstance(per_rid[r], dict)
            and len(per_rid[r]) == 1
            and isinstance(next(iter(per_rid[r].values())), list)
            and len(next(iter(per_rid[r].values()))) == 1
            and torch.is_tensor(next(iter(per_rid[r].values()))[0])
            and next(iter(per_rid[r].values()))[0].is_cuda
            and next(iter(per_rid[r].values()))[0].numel() == 1
            for r in rids
        ):
            keys = {next(iter(per_rid[r].keys())) for r in rids}
            dtypes = {next(iter(per_rid[r].values()))[0].dtype for r in rids}
            if len(keys) == 1 and len(dtypes) == 1:
                uniform_key = next(iter(keys))
        if uniform_key is not None:
            with torch.cuda.stream(side):
                flat_gpu = torch.cat(
                    [next(iter(per_rid[r].values()))[0].reshape(1) for r in rids]
                )
                flat_cpu = self._get_pinned_d2h_buffer(
                    "check_stop_flat", flat_gpu.shape, flat_gpu.dtype,
                )
                flat_cpu.copy_(flat_gpu, non_blocking=True)
            side.synchronize()
            cpu_fast: dict = {
                r: {uniform_key: [flat_cpu[i:i + 1]]}
                for i, r in enumerate(rids)
            }
            return NodeOutput(
                per_request_output_tensors=cpu_fast,
                allocation_failed=output.allocation_failed,
                alloc_pages_short=output.alloc_pages_short,
                alloc_failed_request_id=output.alloc_failed_request_id,
                completion_event=output.completion_event,
            )

        cpu_per_rid: dict = {}
        buffer_indices: dict[tuple[str, torch.dtype, tuple[int, ...]], int] = defaultdict(int)
        with torch.cuda.stream(side):
            for rid, name_to_list in output.per_request_output_tensors.items():
                if not isinstance(name_to_list, dict):
                    cpu_per_rid[rid] = name_to_list
                    continue
                cpu_per_rid[rid] = {}
                for name, tensors in name_to_list.items():
                    if not isinstance(tensors, list):
                        cpu_per_rid[rid][name] = tensors
                        continue
                    new_list = []
                    for t in tensors:
                        if torch.is_tensor(t) and t.is_cuda:
                            key = ("check_stop", t.dtype, tuple(t.shape))
                            idx = buffer_indices[key]
                            buffer_indices[key] += 1
                            cpu_t = self._get_pinned_d2h_buffer(
                                "check_stop", t.shape, t.dtype, idx,
                            )
                            cpu_t.copy_(t, non_blocking=True)
                            new_list.append(cpu_t)
                        else:
                            new_list.append(t)
                    cpu_per_rid[rid][name] = new_list
        side.synchronize()

        return NodeOutput(
            per_request_output_tensors=cpu_per_rid,
            allocation_failed=output.allocation_failed,
            alloc_pages_short=output.alloc_pages_short,
            alloc_failed_request_id=output.alloc_failed_request_id,
            completion_event=output.completion_event,
        )

    def _apply_pending_removes_safe_to_drop(
        self, in_flight_rids: set[str]
    ) -> None:
        """Apply ``REMOVE_REQUEST`` for any rid that is not currently held by
        an in-flight GPU step. Removes for in-flight rids stay deferred and
        are reattempted next iter.

        A rid running on the side stream (MSTAR_SIDE_PREFILL) is also in-flight:
        its GPU work may still be reading/writing that rid's KV pages, so its
        REMOVE must stay deferred until the side batch is postprocessed. We
        union ``self._side_in_flight_rids`` here so every caller respects it
        without having to thread the side set through each call site."""
        held = in_flight_rids | self._side_in_flight_rids
        to_apply = [r for r in self._pending_removes if r not in held]
        for rid in to_apply:
            self._pending_removes.discard(rid)
            self._remove_request(RemoveRequest(request_id=rid, source=MessageSource.SELF))

    def run(self) -> None:
        switch_interval = os.environ.get("MSTAR_PY_SWITCH_INTERVAL_SEC", "")
        if switch_interval:
            try:
                sys.setswitchinterval(float(switch_interval))
                logger.info(
                    "Worker %s: Python thread switch interval set to %ss",
                    self.worker_id,
                    switch_interval,
                )
            except ValueError:
                logger.warning(
                    "Worker %s: ignoring invalid MSTAR_PY_SWITCH_INTERVAL_SEC=%r",
                    self.worker_id,
                    switch_interval,
                )

        # Bound the load-time asymmetry between workers before any
        # subgroup NCCL collective fires inside the per-bs CUDA-graph
        # capture loop. Without this fence, a worker with a small model
        # (e.g. an 8B Talker) can finish loading, enter warmup, and hit
        # its first subgroup barrier while a worker with a 30B Thinker
        # is still streaming safetensors shards. The subgroup NCCL comm
        # is created lazily on that first collective; its connect-retry
        # budget is ~33 s, which is shorter than the load-time delta on
        # large multi-tower models. Syncing here means every worker
        # reaches warmup at the same wall-clock instant, so subgroup
        # bootstrap completes within the retry budget.
        self.tp_groups.barrier_all()

        # CUDA graph capture before entering the main loop
        self.engine_manager.warmup_all()

        # Sync every worker before the main loop opens. Per-batch-size
        # captures inside CudaGraphRunner are already barriered on the
        # node-local TP group, but that doesn't bound the time between
        # ``warmup_and_capture`` returning and ``run()`` starting to
        # schedule. Without this fence, a TP leader can finish warmup
        # quickly, schedule its first batch, and ZMQ-send
        # ``ScheduleTPNode`` to a follower that's still inside another
        # engine's ``warmup``. The follower can't service the message
        # yet, but the leader will sit on the first NCCL collective.
        self.tp_groups.barrier_all()

        # Setup (weight load + warmup + CUDA-graph capture) is complete. Tell
        # the conductor this worker is ready. The conductor blocks its main
        # loop until every worker reports in, so the API server only advertises
        # readiness once all workers can actually serve.
        self.communicator.send(
            "conductor",
            ConductorMessage(
                message_type=ConductorMessageType.SETUP_DONE,
                body=SetupDone(worker_id=self.worker_id),
            ),
        )

        # The async worker path needs decode submission to return quickly so
        # the main loop can overlap queue/tensor polling and post-processing
        # with GPU execution. Run the engine unconditionally on a dedicated
        # 1-worker GPU thread.
        gpu_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"mstar-gpu-{self.worker_id}"
        )
        logger.info(
            "Worker %s: engine runs on dedicated GPU thread",
            self.worker_id,
        )
        # Dedicated thread that pre-plans FlashInfer attention for the
        # speculatively-built next batch. Runs concurrent with main thread's
        # await_gpu (which releases the GIL), so plan()'s Python work isn't
        # contended by main thread's fast/slow postprocess
        #
        # With double-buffered wrappers (CudaGraphRunner.NUM_SLOTS=2) and
        # advance_event signaling, plan(N+1) runs concurrent with replay(N)
        # on the disjoint slot — the actual GPU overlap. plan_executor waits
        # on prev_advance_event (signaled right after advance_seq_lens(N) on
        # the GPU thread, ~tens of µs into replay)
        #
        # Default ON. Set MSTAR_PRE_PLAN_SPEC=0 to fall back to the
        # double-buffer-without-pre-plan baseline.
        pre_plan_spec = os.environ.get("MSTAR_PRE_PLAN_SPEC", "1") == "1"
        plan_executor = None
        if pre_plan_spec:
            plan_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"mstar-plan-{self.worker_id}"
            )
            logger.info(
                "Worker %s: plan_executor enabled — speculative plan() "
                "pre-runs on a dedicated thread",
                self.worker_id,
            )
        # In-flight: (batch, node_batch, batch_partition, future) | None.
        pending: PendingBatch | None = None

        # MSTAR_SIDE_PREFILL: a second 1-worker executor + a separate CUDA
        # stream that runs a prefill/encoder batch CONCURRENTLY with the decode
        # chain. The side executor only EXECUTES pre-built batches; the main
        # thread still owns all scheduling and postprocess. The side stream is
        # created at the least CUDA priority the device supports, so decode
        # replays on the default stream win SM arbitration and prefill fills
        # the gaps (protecting decode ITL). We fall back to a default-priority
        # stream if the priority query is unavailable. Lazily gated so non-CUDA
        # workers and the flag-off path allocate nothing.
        side_executor: ThreadPoolExecutor | None = None
        pending_side: PendingSide | None = None
        if self._side_prefill:
            side_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"mstar-side-{self.worker_id}"
            )
            if torch.cuda.is_available():
                # In CUDA, a MORE-NEGATIVE priority = HIGHER priority; the
                # default stream is priority 0. There is no user priority
                # lower than the default, so the best we can do to keep decode
                # ahead is leave the side stream at default priority and rely
                # on decode being launched first each step. If a wider
                # negative range exists we still keep the side stream at the
                # least-priority (largest) value the device reports.
                side_priority = 0
                try:
                    least, _greatest = torch.cuda.get_stream_priority_range()
                    side_priority = least
                except Exception:
                    side_priority = 0
                try:
                    self._side_stream = torch.cuda.Stream(
                        device=self.device, priority=side_priority
                    )
                except Exception:
                    self._side_stream = torch.cuda.Stream(device=self.device)
            logger.info(
                "Worker %s: MSTAR_SIDE_PREFILL enabled — prefill/encoder "
                "batches run on a side stream concurrent with decode",
                self.worker_id,
            )

        # MSTAR_SPEC_PEEK_FOR_FAIRNESS=1: only break the spec chain when
        # MicroScheduler.has_ready_excluding finds another (node, walk)
        # ready RIGHT NOW. Single-walk workers always speculate; multi-walk
        # workers yield only when there's actual contention.
        max_consecutive_spec = int(os.environ.get("MSTAR_MAX_CONSECUTIVE_SPEC_STEPS", "1024"))
        spec_peek_for_fairness = (
            os.environ.get("MSTAR_SPEC_PEEK_FOR_FAIRNESS", "1") == "1"
        )
        consecutive_spec_steps = 0
        yield_away_from_target: tuple[str, str] | None = None

        def _set_pending(p: PendingBatch):
            nonlocal pending
            pending = p
            self._in_flight_rids = set(p.batch.node_objects.keys()) if p else set()

        # Per-phase wall-clock instrumentation, gated by MSTAR_PHASE_TIMING.
        # When enabled, every Nth speculative iter logs a histogram so we can
        # see whether await_gpu time = "GPU still running" (overlap working)
        # vs "GPU done, idle" (overlap not paying off). Set the env var to a
        # positive integer = the dump period in iters (e.g. 200).
        phase_period = int(os.environ.get("MSTAR_PHASE_TIMING", "0") or "0")
        phase_buf: dict[str, list[float]] = defaultdict(list)
        phase_iter = [0]

        def _phase_record(name: str, dt: float) -> None:
            if phase_period > 0:
                phase_buf[name].append(dt)

        def _phase_flush() -> None:
            if phase_period <= 0 or phase_iter[0] % phase_period != 0:
                return
            samples = sorted((k, v) for k, v in phase_buf.items() if v)
            parts = []
            for name, vs in samples:
                vs.sort()
                n = len(vs)
                p50 = vs[n // 2] * 1000
                p95 = vs[min(n - 1, int(n * 0.95))] * 1000
                mean = (sum(vs) / n) * 1000
                parts.append(f"{name}: p50={p50:.2f}ms p95={p95:.2f}ms mean={mean:.2f}ms n={n}")
            logger.info(
                "Worker %s phase-timing iter=%d: %s",
                self.worker_id, phase_iter[0], " | ".join(parts),
            )
            phase_buf.clear()

        while True:
            from mstar.utils.profiler import range_pop, range_push
            try:
                _iter_start = _time.perf_counter() if phase_period else 0.0
                self._apply_pending_removes_safe_to_drop(
                    self._in_flight_rids
                )

                # 1. CPU preamble — overlaps with GPU(N).
                # synchronize=False on every range so torch.cuda.synchronize()
                # doesn't drain the in-flight GPU work and undo the overlap.
                if self.enable_nvtx:
                    range_push("worker.process_messages", synchronize=False)
                self._process_messages()
                if self.enable_nvtx:
                    range_pop(synchronize=False)

                if self.enable_nvtx:
                    range_push("worker.check_ready_tensors", synchronize=False)
                self._check_ready_tensors()
                if self.enable_nvtx:
                    range_pop(synchronize=False)

                if self.enable_nvtx:
                    range_push("worker.poll_stream_buffers", synchronize=False)
                self._poll_stream_buffers()
                if self.enable_nvtx:
                    range_pop(synchronize=False)

                # 1b. Reap a finished side-stream prefill (MSTAR_SIDE_PREFILL).
                # Opportunistic: never blocks. Postprocess (routing + token
                # emit) runs here on the main thread. No-op when the flag is
                # off or nothing is in flight.
                if self._side_prefill:
                    pending_side = self._reap_side_if_done(pending_side)

                # 2. Speculatively schedule + build N+1 — overlaps with GPU(N).
                # Only when (a) there's a pending step and (b) it's AR-engine.
                # For non-AR or non-loop-body steps, falls through to the
                # non-speculative path below (drain, then schedule).
                speculation = None
                yield_away_from_target = None

                if pending is not None and self._can_speculate(pending.batch):
                    # Fairness check (peek-based, replaces the old iter-
                    # counter cap): only break the spec chain when there's
                    # another (node, walk) actually ready to schedule on
                    # this worker. On single-walk workers (Orpheus LLM,
                    # Orpheus SNAC) this returns False and we always speculate.
                    must_yield_for_fairness = (
                        spec_peek_for_fairness
                        and consecutive_spec_steps >= 1
                        and self.scheduler.has_ready_excluding(
                            self.worker_graphs_manager,
                            (pending.node_name, pending.graph_walk),
                        )
                    )
                    must_yield_away = (
                        consecutive_spec_steps >= max_consecutive_spec
                        or must_yield_for_fairness
                    )
                    if not must_yield_away:
                        if self.enable_nvtx:
                            range_push("worker.speculate", synchronize=False)
                        _t0 = _time.perf_counter() if phase_period else 0.0
                        speculation = self._try_speculate_next(pending)
                        if phase_period:
                            _phase_record("speculate", _time.perf_counter() - _t0)
                        if self.enable_nvtx:
                            range_pop(synchronize=False)
                    if speculation is None:
                        yield_away_from_target = (
                            pending.node_name,
                            pending.graph_walk,
                        ) if must_yield_away else None
                        batch = self.scheduler.get_next_batch(
                            self.worker_graphs_manager,
                            exclude_target=yield_away_from_target,
                        )
                        if batch is not None:
                            node_batch = self._build_node_batch(batch)
                            batch_partition = self.worker_graphs_manager.get_partition_for_node(batch.node_name)
                            logger.debug(f"Yield away: {batch.node_name} {node_batch.request_ids}")
                            speculation = Speculation(
                                scheduled_batch=batch,
                                node_batch=node_batch,
                                consumed_edges=set(),
                                continuing_rids=set(), # n/a
                                partition=batch_partition,
                                is_new_iter=False,
                                is_same_node=False,
                                is_yield_away=True
                            )

                            # send messages to follower ranks if relevant
                            self.maybe_send_zmq_to_tp_followers(node_batch)
                    if speculation is not None:
                        # Reserve the double-buffer slot for batch_(N+1) NOW
                        # so both pre-plan and replay (queued below) target
                        # the SAME slot — and the OPPOSITE slot from
                        # batch_N's in-flight replay. The reservation lives
                        # on spec_node_batch.metadata['cuda_graph_slot'];
                        # the engine forwards it to the runner.
                        if speculation is not None:
                            engine = self.engine_manager.get_engine(
                                speculation.node_batch.node_name
                            )
                            engine.reserve_replay_slot(speculation.node_batch)

                        # Kick off pre-planning on the plan_executor NOW —
                        # its Python work runs while the main thread is in
                        # await_gpu (releases GIL). plan_executor waits on
                        # prev's advance_event so  plan(N+1) starts before
                        # replay(N) finishes. replay(N) keeps running on
                        # the active slot.
                        if speculation is not None and plan_executor is not None:
                            engine = self.engine_manager.get_engine(
                                speculation.node_batch.node_name
                            )
                            prev_advance_event_for_plan: threading.Event | None = None
                            if pending is not None:
                                prev_advance_event_for_plan = (
                                    pending.node_batch.metadata.get("advance_event")
                                )
                            speculation.plan_future = plan_executor.submit(
                                self._pre_plan_for_speculative_batch,
                                engine,
                                speculation.node_batch,
                                prev_advance_event_for_plan,
                            )

                # 3. If pending: await GPU(N), submit speculated GPU(N+1)
                # asap, then post-process N (fast then slow) overlapping
                # with GPU(N+1).
                spec_pending = None
                if pending is not None:
                    if self.enable_nvtx:
                        range_push("worker.await_gpu", synchronize=False)
                    _t0 = _time.perf_counter() if phase_period else 0.0
                    output: NodeOutput = pending.future.result()
                    if phase_period:
                        _phase_record("await_gpu", _time.perf_counter() - _t0)
                    if self.enable_nvtx:
                        range_pop(synchronize=False)

                    # MSTAR_MULTISTEP_DECODE: ``pending`` may have run several
                    # decode steps in one submission. ``output`` is step 1;
                    # ``output.extra_step_outputs`` holds steps 2..N in order.
                    # The NEXT speculation's loop-back token must come from the
                    # LAST step (step N), so thread from ``last_output`` below.
                    # Absent extra steps → last_output IS output (single-step,
                    # byte-identical). allocation_failed is only ever set on a
                    # single-step submission (multistep is gated to
                    # decode-in-cache batches; a mid-sequence OOM raises
                    # AllocationFailedError before any step, yielding one
                    # allocation_failed NodeOutput with no extra steps).
                    last_output: NodeOutput = output
                    if output.extra_step_outputs:
                        last_output = output.extra_step_outputs[-1]

                    # set node._speculatively_scheduled to false, since
                    # the node has just completed
                    for node in pending.batch.node_objects.values():
                        node._speculatively_scheduled = False

                    if output.allocation_failed:
                        # KV-cache OOM on pending. ``_handle_allocation_failure``
                        # offloads or holds the failed rids and pushes their
                        # GraphNodes back to the scheduler queue.
                        self._handle_allocation_failure(
                            pending.batch, pending.node_batch
                        )
                        for node in pending.batch.node_objects.values():
                            node._speculatively_scheduled = False
                        # Speculation cleanup splits by kind:
                        #
                        # * Non-yield-away spec depended on pending's outputs
                        #   (its inputs would be threaded from ``output`` below
                        #   via ``_thread_outputs_to_speculative``). Pending's
                        #   output is invalid, so the spec batch can't run.
                        #
                        # * Yield-away spec is independent of pending.
                        #   But ``_handle_allocation_failure`` may have shifted
                        #   the engine's KV-cache state (paused/offloaded rids),
                        #   so reset pre-plan.
                        if speculation is not None:
                            if speculation.plan_future is not None:
                                speculation.plan_future.result()
                                self._reset_skip_plan_flags(
                                    speculation.node_batch
                                )
                                speculation.plan_future = None
                            if not speculation.is_yield_away:
                                for rid, edges in speculation.consumed_streaming_edges.items():
                                    for edge in edges:
                                        self._return_speculative_streaming_edge(rid, edge)
                                speculation = None

                    if speculation is not None:
                        spec_batch = speculation.scheduled_batch
                        spec_node_batch = speculation.node_batch
                        # Promote per-rid speculative_signals → real inputs.
                        # Feed from the LAST decode step of a multistep pending
                        # (its sampled token is what the next spec step decodes
                        # from); last_output == output for single-step.
                        if not speculation.is_yield_away:
                            self._thread_outputs_to_speculative(
                                speculation, last_output
                            )
                        # set node._speculatively_scheduled to true, so that it doesn't
                        # accidentally get put on the ready queue while already executing
                        for node in spec_batch.node_objects.values():
                            # this does not include the dropped rids
                            node._speculatively_scheduled = True

                        if spec_batch.node_objects:
                            if self.enable_nvtx:
                                range_push("worker.submit_spec", synchronize=False)
                            _t0 = _time.perf_counter() if phase_period else 0.0
                            # If pre-plan was dispatched but the spec_batch
                            # composition changed, fall back to inline planning
                            if speculation.plan_future is not None and speculation.dropped:
                                speculation.plan_future.result()
                                self._reset_skip_plan_flags(speculation.node_batch)
                                speculation.plan_future = None

                            # Attach a fresh advance_event to this batch so
                            # the NEXT iter's plan_executor can gate on
                            # advance_seq_lens(THIS batch).
                            spec_advance_event = threading.Event()
                            spec_node_batch.metadata["advance_event"] = spec_advance_event

                            # Block the main thread until the GPU executor
                            # thread is about to launch CUDA kernels (set
                            # deep in the engine: before graph.replay() in
                            # CudaGraphRunner, or before forward/forward_batched
                            # in the eager AR path).
                            spec_launch_started_event = threading.Event()
                            spec_node_batch.metadata["launch_started_event"] = spec_launch_started_event
                            # MSTAR_MULTISTEP_DECODE: request N inline decode
                            # steps for this submission when the (post-thread)
                            # spec batch is an eligible uniform thinker_decode
                            # step with enough tokens left. Step 1 uses this
                            # batch's pre-plan (opposite slot); step 2+ re-plan
                            # inline on that same slot on the GPU thread. 1 =
                            # unchanged single-step. Gated to same-node loop-back
                            # speculation (a new decode iteration of the SAME
                            # node) so ``speculative_new_iter`` / ``loop_name``
                            # are set — the sequential-step stopped-rid discard
                            # in _postprocess_batch depends on them. Evaluated
                            # after the drop pass above so it sees the final rid
                            # set.
                            spec_multistep = 1
                            if (
                                speculation.is_same_node
                                and not speculation.is_yield_away
                                and speculation.is_new_iter
                            ):
                                spec_multistep = self._eligible_multistep_decode(
                                    spec_batch, spec_node_batch
                                )
                            spec_node_batch.metadata["num_decode_steps"] = spec_multistep
                            spec_future = gpu_executor.submit(
                                self._execute_on_gpu_thread,
                                spec_batch, spec_node_batch,
                                speculation.plan_future,
                                spec_advance_event,
                            )
                            self.wakeup_event.register_future(spec_future)
                            if self.enable_nvtx:
                                range_pop(synchronize=False)
                                range_push("worker.gpu_submit_queued", synchronize=False)
                            spec_launch_started_event.wait(timeout=0.005)
                            if phase_period:
                                _phase_record("submit_spec", _time.perf_counter() - _t0)
                            if self.enable_nvtx:
                                range_pop(synchronize=False)
                            spec_pending = PendingBatch(
                                batch=spec_batch,
                                node_batch=spec_node_batch,
                                node_name=spec_batch.node_name,
                                partition=speculation.partition,
                                graph_walk=spec_batch.graph_walk,
                                future=spec_future,
                                speculative_new_iter=speculation.is_new_iter,
                                loop_name=speculation.loop_name,
                            )
                        elif speculation.plan_future is not None:
                            # All continuing rids were dropped post-thread,
                            # so no spec batch was submitted. Drain the
                            # orphaned pre-plan future and reset the engine's
                            # skip flags so the next plan_attention call
                            # recomputes from scratch instead of trusting
                            # stale wrapper buffers from this aborted spec.
                            speculation.plan_future.result()
                            self._reset_skip_plan_flags(speculation.node_batch)

                    # Post-process N (routing stage) — runs concurrently with
                    # GPU(N+1) if we submitted one above. Skipped on
                    # allocation_failed since the output tensors aren't valid;
                    # ``_handle_allocation_failure`` already rehabilitated the
                    # failed rids upstream.
                    #
                    # MSTAR_MULTISTEP_DECODE: a multistep pending produced one
                    # NodeOutput per decode step (step 1 = ``output``, steps
                    # 2..N = ``output.extra_step_outputs``). Postprocess them
                    # SEQUENTIALLY as N normal steps against the SAME pending:
                    # each call runs mark_node_complete / process_node_outputs /
                    # dynamic_loop_iter_counts / emission / check_stop once,
                    # advancing the Loop's curr_iter one iteration per step, in
                    # order. ``_pending_loop_stops`` set by step k's check_stop
                    # is consumed at step k+1's start (its stopped-rid outputs
                    # discarded, matching the single-step chain) because both
                    # calls share ``self._pending_loop_stops``. Single-step:
                    # the list is just ``[output]`` — byte-identical to before.
                    _t0 = _time.perf_counter() if phase_period else 0.0

                    if not output.allocation_failed:
                        if self.enable_nvtx:
                            range_push("worker.postprocess_batch", synchronize=False)
                        step_outputs = [output]
                        if output.extra_step_outputs:
                            step_outputs.extend(output.extra_step_outputs)
                        for step_out in step_outputs:
                            self._postprocess_batch(pending, step_out)
                        if self.enable_nvtx:
                            range_pop(synchronize=False)

                    # Removes for any rid not in the in-flight spec step
                    # are safe to apply now.
                    in_flight = set(spec_pending.batch.node_objects.keys()) if spec_pending else set()
                    self._apply_pending_removes_safe_to_drop(in_flight)
                    _set_pending(None)

                if spec_pending is not None:
                    if speculation.is_yield_away:
                        consecutive_spec_steps = 0
                    else:
                        consecutive_spec_steps += 1
                    # 3b. Dispatch a prefill/encoder batch onto the side stream
                    # to overlap the decode chain (MSTAR_SIDE_PREFILL). Only
                    # when the chain we just queued is a genuine (non-yield-away)
                    # decode/AR chain — a yield-away step is itself a handoff to
                    # another node, so there's no decode chain to overlap. The
                    # exclude_target keeps the scheduler off the active decode
                    # group; get_next_batch pops the prefill nodes so the
                    # decode speculation next iter won't re-see them.
                    if self._side_prefill and not speculation.is_yield_away:
                        pending_side = self._maybe_dispatch_side(
                            pending_side,
                            side_executor,
                            (spec_pending.node_name, spec_pending.graph_walk),
                        )
                    if phase_period:
                        _phase_record("iter_total", _time.perf_counter() - _iter_start)
                        phase_iter[0] += 1
                        _phase_flush()
                    _set_pending(spec_pending)
                    continue
                consecutive_spec_steps = 0

                # 4. Non-speculative path: no pending or speculation skipped
                # (e.g., non-AR engine, or loop ended). Run MicroScheduler.
                if self.enable_nvtx:
                    range_push("worker.schedule", synchronize=False)
                batch = None
                if yield_away_from_target is not None:
                    batch = self.scheduler.get_next_batch(
                        self.worker_graphs_manager,
                        exclude_target=yield_away_from_target,
                    )
                if batch is None:
                    batch = self.scheduler.get_next_batch(self.worker_graphs_manager)
                if self.enable_nvtx:
                    range_pop(synchronize=False)
                if batch is None:
                    self.communicator.wait_for_work(10)
                    continue

                if self.enable_nvtx:
                    range_push("worker.build_node_batch", synchronize=False)
                node_batch = self._build_node_batch(batch)
                batch_partition = self.worker_graphs_manager.get_partition_for_node(batch.node_name)

                for request_id, req_info in node_batch.per_request_info.items():
                    req_info.dynamic_loop_iter_counts.update(
                        self.worker_graphs_manager.get_dynamic_loop_iters(
                            request_id, partition=batch_partition,
                        )
                    )
                if self.enable_nvtx:
                    range_pop(synchronize=False)

                # Reserve the double-buffer slot on the main thread before
                # submission so the per-key counter advances in main-thread
                # order. Without this, the GPU thread would advance the
                # counter at run time and races with main-thread reservations
                # from later iters.
                fallthrough_engine = self.engine_manager.get_engine(batch.node_name)
                fallthrough_engine.reserve_replay_slot(node_batch)

                # send messages to follower ranks if relevant
                self.maybe_send_zmq_to_tp_followers(node_batch)

                # Attach a fresh advance_event so the next iter's
                # plan_executor (if it speculates) can wait on this batch's
                # advance_seq_lens.
                fallthrough_advance_event = threading.Event()
                node_batch.metadata["advance_event"] = fallthrough_advance_event
                # MSTAR_MULTISTEP_DECODE stays single-step on this
                # (non-speculative) fallthrough path: it's the cold-start /
                # yield-away / drain path, not the steady-state decode chain
                # where cycle amortization pays off, and its PendingBatch lacks
                # the populated ``loop_name`` / ``speculative_new_iter`` that the
                # sequential-step stopped-rid discard relies on. Multistep is
                # applied only on the speculative-submit path above, where both
                # are set. Explicit 1 keeps this path byte-identical.
                node_batch.metadata["num_decode_steps"] = 1
                future = gpu_executor.submit(
                    self._execute_on_gpu_thread, batch, node_batch,
                    None, fallthrough_advance_event,
                )
                self.wakeup_event.register_future(future)
                logger.debug(f"Scheduling: {batch.node_name} {node_batch.request_ids}")
                _set_pending(PendingBatch(
                    batch=batch,
                    node_batch=node_batch,
                    node_name=batch.node_name,
                    partition=batch_partition,
                    graph_walk=batch.graph_walk,
                    future=future
                ))
            except Exception:
                logger.exception("Worker %s error in main loop", self.worker_id)
                sleep(0.01)
