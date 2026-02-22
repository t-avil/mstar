from abc import ABC
from dataclasses import asdict, dataclass
from enum import Enum

from mminf.graph.base import SignalToDests, SignalToDestsAndFlags
from mminf.graph.worker_assignment import Subgraph


class Status(Enum):
    WAITING = "waiting"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    DONE = "done"


@dataclass
class RequestBody(ABC):
    def to_dict(self):
        return asdict(self)
    
    def from_dict(self, input: dict):
        return self(**input)


######################################
# Requests to workers
######################################

class WorkerRequestType(Enum):
    NEW_FWD = "new_fwd"
    REMOVE_REQUEST = "remove_request"
    INPUT_TENSORS = "input_tensors"


@dataclass
class NewFwdRequest(RequestBody):
    request_id: str
    subgraphs: list[Subgraph]
    stage_to_worker: dict[str, str]
    initial_inputs: SignalToDestsAndFlags
    # TODO: actual tensors will be transferred via Ray. This metadata may be
    # transferred via Ray as well, but we are using ZMQ for the sake of
    # initial testing.


@dataclass
class RemoveRequest(RequestBody):
    request_id: str


@dataclass
class InputTensors(RequestBody):
    request_id: str
    inputs: SignalToDests
    # TODO: actual tensors to be transferred via Ray


@dataclass
class WorkerRequest:
    request_type: WorkerRequestType
    request_body: RequestBody


######################################
# Requests to conductor
######################################

class ConductorRequestType(Enum):
    TENSORS = "tensors"
    STAGE_DONE = "stage_done"
    SUBGRAPH_DONE = "subgraph_done"


@dataclass
class ConductorTensors(RequestBody):
    request_id: str
    tensors: SignalToDestsAndFlags


@dataclass
class StageDone(RequestBody):
    request_id: str
    stage_name: str


@dataclass
class SubgraphsDone(RequestBody):
    request_id: str
    subgraph_id: list[str]


@dataclass
class ConductorRequest:
    request_type: ConductorRequestType
    request_body: RequestBody