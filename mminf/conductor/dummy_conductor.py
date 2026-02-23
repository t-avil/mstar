
import zmq

from mminf.conductor.formats import RequestData
from mminf.graph.base import GraphSection, SignalToDestsAndFlags
from mminf.graph.worker_assignment import collect_subgraphs, get_stage_to_worker_id
from mminf.ipc_formats import NewRequest, WorkerMessage, WorkerRequestType


class DummyConductor:
    def __init__(
        self,
        worker_ids: list[str],
        worker_socket_path_prefix: str="/tmp/mminf/workers/",
        conductor_socket_path: str="/tmp/mminf/conductor.ipc"
    ):
        self.requests: dict[str, RequestData] = {}
        self.worker_ids = worker_ids

        self.context = zmq.Context()
        self.result_socket = self.context.socket(zmq.PULL)
        self.result_socket.connect(f"ipc://{conductor_socket_path}")
        self.result_socket.setsockopt(zmq.LINGER, 0)

        self.worker_sockets: dict[str, zmq.SyncSocket] = {}
        for id in worker_ids:
            self.worker_sockets[id] = self.context.socket(zmq.PUSH)
            self.worker_sockets[id].connect(
                f"ipc://{worker_socket_path_prefix}/{id}.ipc"
            )
            self.worker_sockets[id].setsockopt(zmq.LINGER, 0)

    def _process_graph(
        self, request_id: str, graph: GraphSection,
        initial_inputs: SignalToDestsAndFlags,
        output_types: list[str]
    ):
        """
        Given a graph with the worker ids populated for each stage,
        dispatch to workers.
        """

        worker_subgraph = collect_subgraphs(graph)
        stage_to_worker = get_stage_to_worker_id(graph)

        for worker in worker_subgraph:
            self.worker_sockets[worker].send_pyobj(
                WorkerMessage(
                    request_type=WorkerRequestType.NEW_FWD,
                    request_body=NewRequest(
                        request_id=request_id,
                        subgraphs=worker_subgraph[worker],
                        stage_to_worker=stage_to_worker,
                        initial_inputs=initial_inputs
                    )
                )
            )
        
        if request_id not in self.requests:
            self.requests[request_id] = RequestData(
                input_ids=list(initial_inputs.keys()),
                output_types=[], # TODO
            )


        pass