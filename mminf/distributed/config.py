from dataclasses import dataclass


@dataclass
class ShardingGroup:
    nodes: set[str]
    tp_size: int
    graph_walks: set[str] | None = None  # None = all graph walks
    _tp_rank: int | None = None  # set by the conductor

    def __post_init__(self):
        self.nodes = set(self.nodes)
        if self.graph_walks is not None:
            self.graph_walks = set(self.graph_walks)

    def register_workers(self, workers: list[str], my_tp_rank: int | None = None):
        assert len(workers) == self.tp_size, (
            f"register_workers got {len(workers)} workers but tp_size={self.tp_size}"
        )
        self._workers = list(workers)
        self._workers_set = set(workers)
        if my_tp_rank is not None:
            self._tp_rank = my_tp_rank


@dataclass
class ShardDestination:
    worker: str
    full_tensor: bool
    start_idx: int | None = None
    end_idx: int | None = None


NodeAndGraphWalk = tuple[str, str]


@dataclass
class ShardingConfig:
    groups: list[ShardingGroup]
    shard_dim: dict[str, int | None]  # signal name to shard dim (None/absent for replicated)

    def setup(
        self, node_to_worker: dict[NodeAndGraphWalk, list[str]]
    ):
        self.node_to_worker = node_to_worker
        graph_walks = {gw for (_node, gw) in node_to_worker.keys()}
        self.group_mapping: dict[NodeAndGraphWalk, ShardingGroup] = {}
        for group in self.groups:
            gws = group.graph_walks if group.graph_walks is not None else graph_walks
            for node in group.nodes:
                for gw in gws:
                    self.group_mapping[(node, gw)] = group

        for (node, gw) in node_to_worker:
            if (node, gw) in self.group_mapping:
                continue
            singleton = ShardingGroup(
                nodes=[node], tp_size=1, graph_walks=[gw],
            )
            singleton.register_workers(
                list(node_to_worker[(node, gw)]), my_tp_rank=0
            )
            self.group_mapping[(node, gw)] = singleton

    def compute_fanout(
        self, signal: str, source_graph_walk: str,
        source_node: str, dest_node: str,
        shard_dim_size: int,
        dest_graph_walk: str | None = None,
        source_tp_rank: int | None = None,
    ) -> list[ShardDestination]:
        if not hasattr(self, "group_mapping"):
            raise RuntimeError("Must call setup before compute_fanout")

        if dest_graph_walk is None:
            dest_graph_walk = source_graph_walk
        source = (source_node, source_graph_walk)
        dest = (dest_node, dest_graph_walk)
        source_group = self.group_mapping[source]
        dest_group = self.group_mapping[dest]

        shard_dim = self.shard_dim.get(signal)

        if source_tp_rank is None:
            assert source_group._tp_rank is not None, (
                f"source group for {source} has no _tp_rank set; "
                "call register_workers with my_tp_rank, or pass source_tp_rank explicitly"
            )
            source_tp_rank = source_group._tp_rank

        source_worker = source_group._workers[source_tp_rank]
        fanout = []
        if shard_dim is None:  # replicated
            if source_worker in dest_group._workers_set:
                fanout.append(ShardDestination(
                    worker=source_worker, full_tensor=True
                ))
            if source_tp_rank == 0:
                # find all workers that do not already have the tensor
                dest_workers = dest_group._workers_set - source_group._workers_set
                for worker in dest_workers:
                    fanout.append(ShardDestination(
                        worker=worker, full_tensor=True
                    ))
        else:  # sharded
            # like in vLLM, we will enforce that the TP size divides the shard dim size
            total_size = shard_dim_size * source_group.tp_size
            assert total_size % dest_group.tp_size == 0, (
                f"total shard dim size {total_size} not divisible by "
                f"dest tp_size {dest_group.tp_size}"
            )
            dest_shard_size = total_size // dest_group.tp_size

            source_shard_start = source_tp_rank * shard_dim_size
            source_shard_end = source_shard_start + shard_dim_size

            for dest_tp_rank in range(dest_group.tp_size):
                dest_shard_start = dest_tp_rank * dest_shard_size
                dest_shard_end = dest_shard_start + dest_shard_size

                if source_shard_end <= dest_shard_start:
                    break  # beyond the point of overlap, can stop
                if source_shard_start >= dest_shard_end:
                    continue  # not reached the point of overlap yet, keep looking

                fanout.append(ShardDestination(
                    worker=dest_group._workers[dest_tp_rank],
                    full_tensor=False,
                    start_idx=max(source_shard_start, dest_shard_start),
                    end_idx=min(source_shard_end, dest_shard_end)
                ))

        return fanout

    def compute_all_fanouts(
        self, signal: str, source_graph_walk: str,
        source_node: str, dest_node: str,
        shard_dim_size: int,
        dest_graph_walk: str | None = None,
    ) -> dict[str, list[ShardDestination]]:  # source worker -> shard destinations
        source = (source_node, source_graph_walk)
        source_group = self.group_mapping[source]
        fanouts = {}
        for tp_rank in range(source_group.tp_size):
            fanouts[source_group._workers[tp_rank]] = self.compute_fanout(
                signal=signal, source_graph_walk=source_graph_walk,
                source_node=source_node, dest_node=dest_node,
                shard_dim_size=shard_dim_size, dest_graph_walk=dest_graph_walk,
                source_tp_rank=tp_rank
            )
        return fanouts
