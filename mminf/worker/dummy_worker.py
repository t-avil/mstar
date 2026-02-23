from copy import deepcopy
from dataclasses import dataclass, field
import time

import numpy as np
import zmq

from mminf.model.base import Subgraph, TensorData
from mminf.graph.base import GraphPointer, GraphStage, SignalToDests, SignalToDestsAndFlags, remove_flags
from mminf.graph.request_queues import PerRequestStageQueues
from mminf.ipc_formats import (
    ConductorMessage, ConductorRequestType, ConductorTensors, InputTensors,
    NewRequest, RemoveRequest, SubgraphsDone, WorkerMessage, WorkerRequestType
)


@dataclass(frozen=True)
class IdAndPhase:
    id: str
    phase: str


@dataclass
class StageOutputRouting:
    to_conductor: SignalToDestsAndFlags
    to_workers: dict[str, SignalToDests] # worker id to signals
    completed_subgraphs: list[str] = field(default_factory=[])


@dataclass
class SubgraphQueues:
    subgraph_id: str
    phase: str
    subgraph: Subgraph
    per_request_queues: dict[str, PerRequestStageQueues]

    def process_new_inputs(self, request_id: str, inputs: SignalToDests):
        return self.per_request_queues[request_id].process_new_inputs(inputs)
    
    def is_done(self, request_id) -> bool:
        return self.per_request_queues[request_id].waiting is None
    
    def add_request(self, request_id: str):
        self.per_request_queues[request_id] = PerRequestStageQueues(
            waiting=deepcopy(self.subgraph),
            subgraph_id=self.subgraph_id
        )

    def remove_request(self, request_id: str):
        if request_id in self.per_request_queues:
            del self.per_request_queues[request_id]
    
    def get_ready_stage_names(self) -> dict[str, str]:
        # Returns dict of request_id to stage_names
        return {
            request_id: [s.name for s in q.ready] \
                for (request_id, q) in self.per_request_queues.items()
        }

    def pop_ready_stages(
        self, request_id: str, stage_names: list[str]
    ) -> list[GraphStage]:
        stages = []
        if request_id in self.per_request_queues:
            q = self.per_request_queues[request_id]
            pop_idxs = set(
                [i for i, stage in enumerate(q.ready) if stage.name in set(stage_names)]
            )
            stages = [q.ready[i] for i in pop_idxs]
            q.ready = [stage for i, stage in enumerate(q.ready) if i not in pop_idxs]
        return stages
    
    def reset(self, request_id):
        self.per_request_queues[request_id].waiting = deepcopy(self.subgraph)
        self.per_request_queues[request_id].ready = []


@dataclass
class PerRequestInfo:
    stage_to_worker: dict[IdAndPhase, str]
    phase_to_subgraph_ids: dict[str, str]
    current_phase: str = field(default=None)
    tensors: dict[str, TensorData] = field(default_factory=dict)


@dataclass
class SubgraphsManager:
    queues: dict[str, SubgraphQueues] # subgraph_id to queues
    per_request_info: dict[str, PerRequestInfo] # request id to info
    subgraph_id_to_phase: dict[str, str] # for all subgraphs
    all_subgraph_ids_to_stages: dict[str, str]

    def update_phase(self, request_id: str, phase: str):
        self.per_request_info[request_id].current_phase = phase

    def process_new_inputs(
        self,
        request_id: str,
        inputs: SignalToDests
    ):
        phase = self.per_request_info[request_id].current_phase
        subgraph_ids = self.per_request_info[request_id].phase_to_subgraph_ids[phase]
        for subgraph_id in subgraph_ids:
            self.queues[subgraph_id].process_new_inputs(request_id, inputs)

    def process_stage_outputs(
        self, request_id: str,
        outputs: SignalToDestsAndFlags
    ) -> StageOutputRouting:
        phase = self.per_request_info[request_id].current_phase
        # find back_to_conductor flags
        to_conductor = {
            signal: [dest for dest in dests if dest.back_to_conductor] \
                for signal, dests in outputs.items()
        }

        # process all internal-facing outputs
        outputs_no_flags = remove_flags(outputs)
        subgraph_ids = self.per_request_info[request_id].phase_to_subgraph_ids[phase]

        completed_subgraphs = []
        for subgraph_id in subgraph_ids:
            queue = self.queues[subgraph_id]
            outputs_no_flags = queue.process_new_inputs(request_id, outputs_no_flags)
            if queue.is_done(request_id):
                completed_subgraphs.append(subgraph_id)
                queue.reset(request_id)

        # get mapping of worker to proper external outputs
        to_workers: dict[str, SignalToDests] = {}
        for signal, dests in outputs_no_flags.items():
            signal_to_workers: dict[str, list[GraphPointer]] = {} # worker: [graph_pointer]
            for dest in dests:
                worker_id = self.per_request_info[request_id].stage_to_worker[IdAndPhase(
                    stage=dest, phase=phase
                )]
                if worker_id not in signal_to_workers:
                    signal_to_workers[worker_id] = []
                signal_to_workers[worker_id].append(dest)

            for worker_id, pointers in signal_to_workers.items():
                if worker_id not in to_workers:
                    to_workers[worker_id] = {}
                to_workers[worker_id][signal] = pointers

        return StageOutputRouting(
            to_conductor=to_conductor,
            to_workers=to_workers,
            completed_subgraphs=completed_subgraphs
        )
    
    def add_request(
        self, request_id: str,
        subgraph_ids: list[str], # for our subgraphs
        subgraph_to_worker: dict[str, str] # for other / all subgraphs
    ):
        stage_to_worker = {}
        phase_to_subgraph_ids: dict[str, list[str]] = {}
        for id in subgraph_ids:
            phase = self.queues[id].subgraph.phase
            if phase not in phase_to_subgraph_ids:
                phase_to_subgraph_ids[phase] = []
            phase_to_subgraph_ids[phase].append(id)
    
            self.queues[id].add_request(request_id)
        
        for subgraph_id, worker_id in subgraph_to_worker.items():
            phase = self.subgraph_id_to_phase[subgraph_id]
            stage_to_worker.update({
                IdAndPhase(
                    id=name,
                    phase=phase
                ): worker_id for name in self.all_subgraph_ids_to_stages[subgraph_id]
            })
        self.per_request_info[request_id] = PerRequestInfo(
            stage_to_worker=stage_to_worker,
            phase_to_subgraph_ids=phase_to_subgraph_ids
        )
    
    def remove_request(self, request_id: str):
        if request_id in self.per_request_info:
            for ids in self.per_request_info[request_id].phase_to_subgraph_ids.values():
                for queue_id in ids:
                    self.queues[queue_id].remove_request(request_id)
            del self.per_request_info[request_id]


class DummyWorker:
    def __init__(
        self,
        worker_id: str,
        worker_ids: list[str],
        my_subgraphs: list[Subgraph],
        subgraph_id_to_phase: dict[str, str], # for all subgraphs
        subgraph_id_to_stages: dict[str, list[str]], # for all subgraphs
        worker_socket_path_prefix: str="/tmp/mminf/workers/",
        conductor_socket_path: str="/tmp/mminf/conductor.ipc"
    ):
        self.worker_id = worker_id
        self.subgraphs_manager = SubgraphsManager(
            queues={
                subgraph.subgraph_id: SubgraphQueues(
                    subgraph_id=subgraph.subgraph_id,
                    phase=subgraph.phase,
                    subgraph=subgraph,
                    per_request_queues={}
                ) for subgraph in my_subgraphs
            },
            per_request_info={},
            subgraph_id_to_phase=subgraph_id_to_phase,
            all_subgraph_ids_to_stages=subgraph_id_to_stages
        )

        # set up sockets between the other workers
        self.context = zmq.Context()
        self.request_socket = self.context.socket(zmq.PULL)
        self.request_socket.bind(f"ipc://{worker_socket_path_prefix}/{worker_id}.ipc")
        self.request_socket.setsockopt(zmq.LINGER, 0)

        self.result_socket = self.context.socket(zmq.PUSH)
        self.result_socket.connect(f"ipc://{conductor_socket_path}")
        self.result_socket.setsockopt(zmq.LINGER, 0)

        self.inter_worker_sockets: dict[str, zmq.SyncSocket] = {}
        for id in worker_ids:
            if id == worker_id:
                continue
            self.inter_worker_sockets[id] = self.context.socket(zmq.PUSH)
            self.inter_worker_sockets[id].connect(
                f"ipc://{worker_socket_path_prefix}/{id}.ipc"
            )
            self.inter_worker_sockets[id].setsockopt(zmq.LINGER, 0)
        
    def _ingest_request(
        self, body: NewRequest
    ):
        self.subgraphs_manager.add_request(
            request_id=body.request_id,
            subgraph_ids=body.subgraph_ids,
            subgraph_to_worker_id=body.subgraph_to_worker
        )
        self.subgraphs_manager.update_phase(
            body.request_id, body.initial_phase
        )
        self.subgraphs_manager.process_new_inputs(
            request_id=body.request_id,
            inputs=remove_flags(body.initial_inputs)
        )

    def _remove_request(self, body: RemoveRequest):
        self.subgraphs_manager.remove_request(body.request_id)
    
    def _process_new_inputs(
        self, body: InputTensors
    ):
        self.subgraphs_manager.update_phase(
            body.request_id, body.phase
        )
        self.subgraphs_manager.process_new_inputs(
            request_id=body.request_id,
            inputs=remove_flags(body.inputs)
        )

    def _process_requests(self):
        """
        Processes all pending requests
        """
        while True:
            try:
                request: WorkerMessage = self.request_socket.recv_pyobj(
                    flags=zmq.NOBLOCK
                )
                if request.request_type == WorkerRequestType.NEW_REQUEST:
                    self._ingest_request(request.request_body)
                elif request.request_type == WorkerRequestType.REMOVE_REQUEST:
                    self._remove_request(request.request_body)
                elif request.request_type == WorkerRequestType.INPUT_TENSORS:
                    self._process_new_inputs(request.request_body) 
            except zmq.Again:
                break

    def _send_outputs(self, request_id: str, outputs: StageOutputRouting):
        # to workers
        for worker in outputs.to_workers:
            request = WorkerMessage(
                request_type=WorkerRequestType.INPUT_TENSORS,
                request_body=InputTensors(
                    request_id=request_id,
                    inputs=outputs.to_workers[worker]
                )
            )
            self.inter_worker_sockets[worker].send_pyobj(request)
        
        # to conductor
        if outputs.to_conductor:
            request = ConductorMessage(
                request_type=ConductorRequestType.TENSORS,
                request_body=ConductorTensors(
                    request_id=request_id,
                    inputs=outputs.to_conductor
                )
            )
            self.result_socket.send_pyobj(request)
        
        if outputs.completed_subgraphs:
            request = ConductorMessage(
                request_type=ConductorRequestType.SUBGRAPHS_DONE,
                request_body=SubgraphsDone(
                    request_id=request_id,
                    subgraph_ids=outputs.completed_subgraphs
                )
            )
        
    def run(self):
        # TODO: this is just a dummy version
        while True:
            self._process_requests()
            for queue in self.subgraphs_manager.queues.values():
                ready_stage_names = queue.get_ready_stage_names()

                for request_id, names in ready_stage_names.items():
                    stages = queue.pop_ready_stages(request_id, names)
                    for s in stages:
                        outputs = self.subgraphs_manager.process_stage_outputs(
                            request_id, s.outputs
                        )
                        self._send_outputs(request_id, outputs)
            time.sleep(np.random.rand() * 2)