from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

try:
    import torch.distributed._symmetric_memory as symm_mem
    _HAS_SYMM_MEM = True
except Exception:  # pragma: no cover - older torch
    symm_mem = None
    _HAS_SYMM_MEM = False

# Low-latency TP=2 all-reduce via PyTorch symmetric memory (the capturable,
# NVLink one-shot/two-shot reduction vLLM uses instead of NCCL ring). one_shot
# below the threshold, two_shot above (matches torch-inductor + vLLM ws=2).
# Off by default; enable with MSTAR_SYMM_ALLREDUCE=1.
_SYMM_ONE_SHOT_MAX_BYTES = 128 * 1024
_SYMM_DTYPE = torch.bfloat16
# Flat workspace element count (bf16): 32M elems = 64 MiB, matches vLLM's ws=2
# cap; any all-reduce tensor larger than this falls back to NCCL.
_SYMM_MAX_NUMEL = 32 * 1024 * 1024
# group_name -> flat rendezvous'd symm buffer (module-level => static address,
# never freed, so it satisfies the CUDA-graph static-buffer contract).
_SYMM_BY_GROUP: dict[str, "torch.Tensor"] = {}


def _symm_allreduce_enabled() -> bool:
    import os
    return _HAS_SYMM_MEM and os.environ.get("MSTAR_SYMM_ALLREDUCE", "0") == "1"


class TPCommGroup:
    def __init__(
        self,
        my_global_rank: int,
        my_group_rank: int,
        group_members: list[int]
    ):
        self.global_rank = my_global_rank
        self.rank = my_group_rank
        self.group_members = group_members
        self.world_size = len(group_members)
        self.device_group = None
        self.initialized = False
        # Rendezvous'd symmetric-memory workspace for the fast all-reduce path
        # (set in init_dist for TP=2 groups; None => NCCL fallback).
        self.symm_buffer: "torch.Tensor | None" = None
        self.symm_group_name: "str | None" = None

    @classmethod
    def trivial(cls) -> "TPCommGroup":
        """A degenerate single-rank group. All collectives are no-ops;
        ``init_process_group`` does nothing. Useful as the default for
        non-TP runs so the same code path works everywhere."""
        return cls(my_global_rank=0, my_group_rank=0, group_members=[0])

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        if dim < 0:
            # Convert negative dim to positive
            dim += input_.dim()
        input_size = input_.size()
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        # Allocate output tensor
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        # All-gather
        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
        # Reshape
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
        return output_tensor

    def barrier(self):
        if self.world_size == 1:
            return
        dist.barrier(group=self.device_group)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        buf = self.symm_buffer
        if buf is not None and self.world_size == 2:
            n = input_.numel()
            nbytes = n * input_.element_size()
            if (
                input_.is_cuda
                and input_.is_contiguous()
                and input_.dtype == buf.dtype
                and n <= buf.numel()
                and nbytes % 4 == 0  # uint32 signal-word alignment
            ):
                slot = buf[:n]          # offset 0 => 16B-aligned symm slice
                flat = input_.view(-1)  # contiguous alias of input_ (in-place contract)
                slot.copy_(flat)
                if nbytes <= _SYMM_ONE_SHOT_MAX_BYTES:
                    # one-shot: pull peers + reduce, write straight back into input_
                    torch.ops.symm_mem.one_shot_all_reduce_out(
                        slot, "sum", self.symm_group_name, flat
                    )
                elif n % self.world_size == 0:
                    # two-shot: reduce-scatter + all-gather, in-place on symm buf
                    torch.ops.symm_mem.two_shot_all_reduce_(
                        slot, "sum", self.symm_group_name
                    )
                    flat.copy_(slot)
                else:
                    dist.all_reduce(input_, group=self.device_group)
                return input_
        dist.all_reduce(input_, group=self.device_group)
        return input_

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output_tensor = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        # Perform reduce-scatter operation
        dist.reduce_scatter_tensor(
            output_tensor, input_tensor, group=self.device_group
        )

        # Reshape before returning
        return output_tensor.movedim(0, dim).contiguous()

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast a tensor from source rank to all ranks."""
        if self.world_size == 1:
            return tensor
        dist.broadcast(tensor, self.group_members[src], self.device_group)
        return tensor


@dataclass
class WorkerTPGroups:
    num_workers: int
    global_rank: int
    # True iff any worker in the run uses TP. Set by GlobalTPConfig from
    # the global worker-graph view so all ranks agree.
    any_tp: bool = False
    # Every distinct TP rank tuple in the run, sorted for stable iteration
    # order across workers. Set by GlobalTPConfig. ``init_dist`` calls
    # ``dist.new_group`` once per entry on every rank — including ranks
    # that aren't members of the group — because PyTorch assigns an
    # auto-incrementing tag inside ``new_group`` that all ranks must agree
    # on; asymmetric call counts deadlock the participating ranks.
    world_tp_groups: list[tuple[int, ...]] = field(default_factory=list)
    node_to_group: dict[str, TPCommGroup] = field(default_factory=dict)

    def add(self, node: str, comm_group: TPCommGroup):
        # disallow colocation of multiple comm groups on the same node
        if node in self.node_to_group and self.node_to_group[node].group_members != comm_group.group_members:
            raise RuntimeError(
                f"Node {node} already has a comm group assigned for worker {self.global_rank}"
            )
        if node not in self.node_to_group:
            self.node_to_group[node] = comm_group

    def init_dist(
        self, init_method="tcp://127.0.0.1:29500",
    ):
        """Initialize the NCCL world group and per-node TP subgroups.

        Every worker calls ``dist.init_process_group`` when *any* worker
        in the run participates in TP (``self.any_tp``) — otherwise ranks
        with no local TP would skip the call and the TP-participating
        ranks would hang waiting for them.

        Subgroup creation: PyTorch's ``dist.new_group`` is collective on
        the global world. It assigns an auto-incrementing tag inside the
        call that every rank must agree on; if non-member ranks skip the
        call, the tag counter drifts and member ranks deadlock. We
        therefore call ``new_group`` once per distinct TP rank tuple on
        every rank — members keep the returned handle, non-members
        discard it.
        """
        torch.cuda.set_device(self.global_rank)
        if not self.any_tp:
            return

        # Pin the device for this rank so NCCL does not "guess device ID based
        # on global rank" — with CUDA_VISIBLE_DEVICES remapping (e.g. 4,5,6,7)
        # that guess is heterogeneous and deadlocks the first collective. The
        # visible-device index for this rank equals global_rank (set_device above).
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            world_size=self.num_workers,
            rank=self.global_rank,
            device_id=torch.device("cuda", self.global_rank),
        )

        rank_tuple_to_pg: dict[tuple[int, ...], "dist.ProcessGroup"] = {}
        for rank_tuple in self.world_tp_groups:
            rank_tuple_to_pg[rank_tuple] = dist.new_group(ranks=list(rank_tuple))

        # Prime every TP subgroup this rank belongs to with one dummy all-reduce
        # BEFORE weight load and CUDA-graph capture. ``dist.new_group`` is lazy
        # for NCCL: the subgroup communicator is created — and its cross-rank
        # connections established via a host-side rendezvous — only on the
        # subgroup's FIRST collective. That first-collective bootstrap must NOT
        # happen inside a CUDA-graph capture (it is not capturable and can hang /
        # trip the NCCL watchdog, rethrown at ProcessGroupNCCL.cpp:2063, killing
        # the worker). It also must not happen at a moment when the two TP ranks
        # are far apart in wall-clock time (one still loading a 30B tower), or the
        # subgroup connect-retry budget (~33 s) is exceeded. Here — right after
        # ``init_process_group``'s global rendezvous — all ranks are tightly
        # synchronized, so priming now forces every subgroup comm to fully
        # connect while its members are co-located in time. The in-capture
        # all-reduces (ParallelAttention o_proj, ParallelSparseMoeBlock) then
        # only RECORD kernels onto an already-connected comm.
        #
        # Iterate the globally-sorted ``world_tp_groups`` so every member primes
        # each group in the same order; a rank skips groups it is not a member of
        # (calling a collective on a non-member group is illegal). No-op when
        # there are no multi-rank groups.
        if self.world_tp_groups:
            prime_device = torch.device("cuda", self.global_rank)
            for rank_tuple in self.world_tp_groups:
                if self.global_rank not in rank_tuple:
                    continue
                dummy = torch.zeros(1, device=prime_device)
                dist.all_reduce(dummy, group=rank_tuple_to_pg[rank_tuple])
            torch.cuda.synchronize()
            # Global fence so no rank races ahead into weight load / capture
            # while a peer is still finishing its subgroup bootstrap.
            dist.barrier()

            # ---- Symmetric-memory low-latency all-reduce (TP=2 fast path) ----
            # Allocate one flat symm-mem workspace per TP=2 subgroup and
            # ``rendezvous`` it NOW, in the same co-timed window as NCCL priming
            # (rendezvous is a collective host-side handshake, NOT capturable —
            # the per-call copy + *_all_reduce ops ARE). Only members call it;
            # any failure leaves the group on the NCCL path (correctness-safe).
            if _symm_allreduce_enabled():
                for rank_tuple in self.world_tp_groups:
                    if self.global_rank not in rank_tuple:
                        continue
                    pg = rank_tuple_to_pg[rank_tuple]
                    if pg.size() != 2:
                        continue
                    gname = pg.group_name
                    if gname in _SYMM_BY_GROUP:
                        continue
                    try:
                        b = symm_mem.empty(
                            _SYMM_MAX_NUMEL, dtype=_SYMM_DTYPE,
                            device=torch.device("cuda", self.global_rank),
                        )
                        if symm_mem.rendezvous(b, pg) is None:
                            raise RuntimeError("symm rendezvous returned None")
                        _SYMM_BY_GROUP[gname] = b
                    except Exception:
                        _SYMM_BY_GROUP.pop(gname, None)
                torch.cuda.synchronize()
                dist.barrier()

        seen: set[int] = set()
        for comm_group in self.node_to_group.values():
            if id(comm_group) in seen:
                continue
            seen.add(id(comm_group))
            if comm_group.world_size == 1:
                comm_group.initialized = True
                continue
            comm_group.device_group = rank_tuple_to_pg[tuple(comm_group.group_members)]
            comm_group.symm_group_name = comm_group.device_group.group_name
            comm_group.symm_buffer = _SYMM_BY_GROUP.get(comm_group.symm_group_name)
            comm_group.initialized = True

    def get_tp_config_for_node(self, node: str) -> TPCommGroup:
        if node not in self.node_to_group:
            self.node_to_group[node] = TPCommGroup.trivial()
        return self.node_to_group[node]

    def barrier_all(self) -> None:
        """Global barrier across every worker process in the run.

        No-op when ``any_tp`` is False (no NCCL world was initialized in
        ``init_dist``). Otherwise calls ``dist.barrier()`` on the default
        global process group, syncing both TP-participating and non-TP
        workers. Used at phase boundaries that require all ranks to be
        ready — e.g. between CUDA-graph warmup and the worker's main
        loop, so a TP leader can't send a ``ScheduleTPNode`` to a
        follower that's still inside ``engine.warmup``.
        """
        if not self.any_tp:
            return
        dist.barrier()


class GlobalTPConfig:
    def __init__(
        # leaving type annotation as Any due to circular import
        self, worker_graphs: dict[str, Any],
        worker_ids: list[str]
    ):
        self.num_workers = len(worker_ids)
        any_tp = any(wg.tp_size > 1 for wg in worker_graphs.values())
        world_tp_groups: list[tuple[int, ...]] = sorted({
            tuple(rank_group)
            for wg in worker_graphs.values()
            for rank_group in wg._tp_ranks
            if len(rank_group) > 1
        })
        self.per_worker_config: dict[str, WorkerTPGroups] = {
            wid: WorkerTPGroups(
                global_rank=i, num_workers=self.num_workers,
                any_tp=any_tp,
                world_tp_groups=world_tp_groups,
            ) for i, wid in enumerate(worker_ids)
        }

        # (global rank, (group ranks...)) -> comm group
        self.comm_groups: dict[tuple[int, tuple], TPCommGroup] = {}
        for wg in worker_graphs.values():
            for rank_group in wg._tp_ranks:
                rank_group_tuple = tuple(rank_group)
                for i, rank in enumerate(rank_group):
                    key = (rank, rank_group_tuple)
                    if key not in self.comm_groups:
                        self.comm_groups[key] = TPCommGroup(
                            my_global_rank=rank,
                            my_group_rank=i,
                            group_members=rank_group
                        )
                    for node in wg.section.get_nodes().keys():
                        self.per_worker_config[worker_ids[rank]].add(
                            node,  self.comm_groups[key]
                        )

