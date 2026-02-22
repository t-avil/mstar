from dataclasses import dataclass, field

import zmq

from mminf.conductor.formats import TensorData
from mminf.graph.base import GraphPointer, GraphStage, SignalToDests, SignalToDestsAndFlags, remove_flags
from mminf.graph.request_queues import RequestQueues
from mminf.graph.worker_assignment import Subgraph


@dataclass
class StageOutputRouting:
    to_conductor: SignalToDestsAndFlags
    to_workers: dict[str, SignalToDestsAndFlags] # worker id to signals


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


    def process_init_inputs(self, initial_inputs: SignalToDestsAndFlags):
        initial_inputs = remove_flags(initial_inputs)
        for queue in self.queues:
            initial_inputs = queue.process_new_inputs(initial_inputs)

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
        for q in self.queues:
            outputs_no_flags = q.process_new_inputs(outputs_no_flags)

        # get mapping of worker to proper external outputs
        to_workers: dict[str, SignalToDestsAndFlags] = {}
        for signal, dests in outputs_no_flags.items():
            signal_to_workers: dict[str, list[GraphPointer]] = {} # worker: [graph_pointer]
            for dest in dests:
                worker_id = self.stage_to_worker[dest]
                if worker_id not in signal_to_workers:
                    signal_to_workers[worker_id] = []
                signal_to_workers[worker_id].append(GraphPointer(dest))
            for worker_id, pointers in signal_to_workers.items():
                if worker_id not in to_workers:
                    to_workers[worker_id] = {}
                to_workers[worker_id][signal] = pointers

        return StageOutputRouting(
            to_conductor=to_conductor,
            to_workers=to_workers
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
        self, request_id: str,
        subgraphs: list[Subgraph],
        stage_to_worker: dict[str, str],
        initial_inputs: SignalToDestsAndFlags,
        # initial_tensors: dict[str, TensorData],
    ):
        queues = [RequestQueues(s) for s in subgraphs]
        stage_to_queue_idx = {}
        for i, s in enumerate(subgraphs):
            stage_to_queue_idx.update({
                stage: i for stage in s.section.get_stage_names()
            })
        self.request_managers[request_id] = RequestManager(
            queues=queues,
            stage_to_queue_idx=stage_to_queue_idx,
            stage_to_worker=stage_to_worker
        )
        self.request_managers[request_id].process_init_inputs(initial_inputs)

    
    def _remove_request(self, request_id: str):
        del self.request_managers[request_id]
    
    