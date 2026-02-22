from dataclasses import dataclass, field
import time

import numpy as np
import zmq

from mminf.conductor.formats import TensorData
from mminf.graph.base import GraphPointer, GraphStage, SignalToDests, SignalToDestsAndFlags, remove_flags
from mminf.graph.request_queues import RequestQueues
from mminf.graph.worker_assignment import Subgraph
from mminf.ipc_formats import ConductorRequest, ConductorRequestType, ConductorTensors, InputTensors, NewFwdRequest, RemoveRequest, SubgraphsDone, WorkerRequest, WorkerRequestType


@dataclass
class StageOutputRouting:
    to_conductor: SignalToDestsAndFlags
    to_workers: dict[str, SignalToDests] # worker id to signals
    completed_subgraphs: list[str] = field(default_factory=[])


# TODO note: need to figure out how this works for producer-consumer streams;
# maybe we can add a new queue to the RequestManager.queues every time the
# stream consumes enough inputs to kick off

@dataclass
class RequestManager:
    queues: list[RequestQueues] # one for each subgraph
    stage_to_queue_idx: dict[str, int] # for stages processed by this worker
    stage_to_worker: dict[str, str] # for all stages
    # execution_strategy: ExecutionStrategy # not implemented yet
    tensors: dict[str, TensorData] = field(default_factory=dict)


    def process_new_inputs(self, inputs: SignalToDests):
        inputs = inputs
        for queue in self.queues:
            inputs = queue.process_new_inputs(inputs)

    def get_ready_stage_names(self) -> list[str]:
        return sum(
            [s.name for s in q.ready] for q in self.queues
        )
    
    def pop_ready_stage(self, stage_name) -> GraphStage:
        for q in self.queues:
            for i, stage in enumerate(q.ready):
                if stage.name ==stage_name:
                    return q.ready.pop(i)
    
    def pop_ready_stages(self, stage_names: list[str]) -> list[GraphStage]:
        return [self.pop_ready_stage(n) for n in stage_names]

    def process_stage_outputs(
        self, outputs: SignalToDestsAndFlags
    ) -> StageOutputRouting:
        # find back_to_conductor flags
        to_conductor = {
            signal: [dest for dest in dests if dest.back_to_conductor] \
                for signal, dests in outputs.items()
        }

        # process all internal-facing outputs
        outputs_no_flags = remove_flags(outputs)

        done_queue_idxs = []
        completed_subgraphs = []
        for i, q in enumerate(self.queues):
            outputs_no_flags = q.process_new_inputs(outputs_no_flags)
            if q.waiting is None: # done with subgraph
                done_queue_idxs.append(i)
                completed_subgraphs.append(q.subgraph_id)
        self.queues = [q for i, q in enumerate(self.queues) if i not in done_queue_idxs]

        # get mapping of worker to proper external outputs
        to_workers: dict[str, SignalToDests] = {}
        for signal, dests in outputs_no_flags.items():
            signal_to_workers: dict[str, list[GraphPointer]] = {} # worker: [graph_pointer]
            for dest in dests:
                worker_id = self.stage_to_worker[dest]
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


class DummyWorker:
    def __init__(
        self,
        worker_id: str,
        worker_ids: list[str],
        worker_socket_path_prefix: str="/tmp/mminf/workers/",
        conductor_socket_path: str="/tmp/mminf/conductor.ipc"
    ):
        self.worker_id = worker_id
        self.request_managers: dict[str, RequestManager] = {}

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
        self, body: NewFwdRequest
    ):
        queues = [RequestQueues(s, subgraph_id=s.subgraph_id) for s in body.subgraphs]
        stage_to_queue_idx = {}
        for i, s in enumerate(body.subgraphs):
            stage_to_queue_idx.update({
                stage: i for stage in s.section.get_stage_names()
            })
        self.request_managers[body.request_id] = RequestManager(
            queues=queues,
            stage_to_queue_idx=stage_to_queue_idx,
            stage_to_worker=body.stage_to_worker
        )
        self.request_managers[body.request_id].process_new_inputs(
            remove_flags(body.initial_inputs)
        )

    def _remove_request(self, body: RemoveRequest):
        if body.request_id in self.request_managers:
            del self.request_managers[body.request_id]
    
    def _process_new_inputs(
        self, body: InputTensors
    ):
        if body.request_id in self.request_managers:
            self.request_managers[body.request_id].process_new_inputs(body.inputs)
    

    def _process_requests(self):
        """
        Processes all pending requests
        """
        while True:
            try:
                request: WorkerRequest = self.request_socket.recv_pyobj(
                    flags=zmq.NOBLOCK
                )
                if request.request_type == WorkerRequestType.NEW_FWD:
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
            request = WorkerRequest(
                request_type=WorkerRequestType.INPUT_TENSORS,
                request_body=InputTensors(
                    request_id=request_id,
                    inputs=outputs.to_workers[worker]
                )
            )
            self.inter_worker_sockets[worker].send_pyobj(request)
        
        # to conductor
        if outputs.to_conductor:
            request = ConductorRequest(
                request_type=ConductorRequestType.TENSORS,
                request_body=ConductorTensors(
                    request_id=request_id,
                    inputs=outputs.to_conductor
                )
            )
            self.result_socket.send_pyobj(request)
        
        if outputs.completed_subgraphs:
            request = ConductorRequest(
                request_type=ConductorRequestType.SUBGRAPHS_DONE,
                request_body=SubgraphsDone(
                    request_id=request_id,
                    subgraph_ids=outputs.completed_subgraphs
                )
            )
        
    def run(self):
        while True:
            self._process_requests()
            for request_id, manager in self.request_managers.items():
                ready_stage_names = manager.get_ready_stage_names()
                if not ready_stage_names:
                    continue
                stages = manager.pop_ready_stages(ready_stage_names)
                # TODO actually execute the stages and get real outputs
                for s in stages:
                    outputs = manager.process_stage_outputs(s.outputs)
                    self._send_outputs(request_id, outputs)
            time.sleep(np.random.rand() * 2)