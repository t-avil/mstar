from dataclasses import dataclass

import torch

from mminf.graph.base import GraphSection
from mminf.graph.worker_assignment import Subgraph


@dataclass
class RequestData:
    input_ids: list[str]
    output_types: list[str]
    subgraphs: dict[str, list[Subgraph]] # subgraphs assigned to each worker
    all_subgraph_ids: set[str]
    completed_subgraph_ids: set[str]
    passed_back_tensor_ids: list[str] # TODO: this should be actual tensors
    is_prefill: bool

    # TODO: will need to add to this as we build things out


@dataclass
class TensorData:
    tensor: torch.Tensor

    # list of segment boundaries (e.g., [(0, 10), (50, 100)] means tokens
    # 0 (inclusive) to 10 (exclusive) and 50 to 100.
    token_ranges: list[tuple[int, int]]