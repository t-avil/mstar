from abc import ABC
from dataclasses import asdict, dataclass
from enum import Enum

from mminf.graph.base import SignalToDestsAndFlags
from mminf.graph.worker_assignment import Subgraph


class Status(Enum):
    WAITING = "waiting"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    DONE = "done"



class WorkerRequestType(Enum):
    NEW_FWD = "new_fwd"
    REMOVE_REQUEST = "remove_request"
    INPUT_TENSORS = "input_tensors"


@dataclass
class RequestBody(ABC):
    def to_dict(self):
        return asdict(self)
    
    def from_dict(self, input: dict):
        return self(**input)


@dataclass
class NewFwdRequest:
    request_id: str
    subgraphs: list[Subgraph]
    stage_to_worker: dict[str, str]
    initial_inputs: SignalToDestsAndFlags


@dataclass
class WorkerRequest:
    request_type: str
    request_body: dict