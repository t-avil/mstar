from dataclasses import dataclass, field
from enum import Enum
from typing import Type
from uuid import uuid4

from mminf.graph.base import GraphSection, GraphStage, Loop, Parallel, Sequential
from mminf.ipc_formats import Status


@dataclass
class Subgraph:
    section: GraphSection
    consumes_stream: bool = field(default=False)
    worker_id: str | None = field(default=None)
    status: Status = field(default=Status.WAITING)
    subgraph_id: str = field(default_factory=lambda: str(uuid4()))


def _combine_sections_sequential_or_parallel(
    section: GraphSection, other: GraphSection,
    comb_type: Type[Sequential] | Type[Parallel]
):
    if isinstance(section, comb_type) and isinstance(other, comb_type):
        section.sections.extend(other.sections)
        return section
    if isinstance(section, comb_type):
        section.sections.append(other)
        return section
    if isinstance(other, comb_type):
        other.sections.insert(0, section)
        return other
    return comb_type([section, other])


def _make_subgraphs(
    graph: GraphSection
) -> list[Subgraph]:
    if isinstance(graph, GraphStage):
        assert graph.worker_id is not None
        return [Subgraph(
            section=graph,
            consumes_stream=graph.consumes_stream,
            worker_id=graph.worker_id
        )]

    if isinstance(graph, Sequential):
        subgraphs = _make_subgraphs(graph.sections[0])

        for i in range(1, len(graph.sections)):
            # Go through it sequentially and merge adjacent sections
            # that are on the same device
            new_subgraphs = _make_subgraphs(graph.sections[i])
            if new_subgraphs[0].worker_id == subgraphs[-1].worker_id and \
                    not new_subgraphs[0].consumes_stream:
                subgraphs[-1].section = _combine_sections_sequential_or_parallel(
                    subgraphs[-1].section, new_subgraphs.pop(0).section,
                    comb_type=Sequential
                )
            subgraphs.extend(new_subgraphs)
        return subgraphs
    
    if isinstance(graph, Parallel):
        all_subgraphs = [
            _make_subgraphs(s) for s in graph.sections
        ]
        # parallel sections that are all on the same worker can be merged
        singleton_subgraphs = [
            s[0] for s in all_subgraphs if len(s) == 1 and not s[0].consumes_stream
        ]
        worker_to_subgraph = {}
        for s in singleton_subgraphs:
            if s.worker_id in worker_to_subgraph:
                worker_to_subgraph[s.worker_id] = _combine_sections_sequential_or_parallel(
                    worker_to_subgraph[s.worker_id], s.section,
                    comb_type=Parallel
                )
            else:
                worker_to_subgraph[s.worker_id] = s

        return list(worker_to_subgraph.values()) + sum([
            s for s in all_subgraphs if len(s) > 1 or s[0].consumes_stream
        ], start=[]) # remaining subgraphs
        
    if isinstance(graph, Loop):
        loop_section_subgraphs = _make_subgraphs(graph.section)
        if len(loop_section_subgraphs) == 1:
            # fully colocated case
            loop_section_subgraphs[0].section = graph
            return loop_section_subgraphs
        
        # in the disaggregated case, we need to wrap all subgraphs in a loop
        # with the external signals and loop-back signals pre-computed
        for s in loop_section_subgraphs:
            s.section = Loop(
                section=s.section,
                n_iters=graph.n_iters,
                curr_iter=graph.curr_iter,
                external_inputs=graph.external_inputs,
                loop_back_signals=graph.loop_back_signals,
                outputs=graph.outputs
            )
        return loop_section_subgraphs


def collect_subgraphs(
    graph: GraphSection
) -> dict[str, list[Subgraph]]:
    """
    Produces a mapping of worker id to list of subgraphs
    """
    subgraphs = _make_subgraphs(graph)
    res: dict[str, list[Subgraph]] = {}
    for s in subgraphs:
        if s.worker_id not in res:
            res[s.worker_id] = []
        res[s.worker_id].append(s)
    return res


def get_stage_to_worker_id(
    graph: GraphSection
) -> dict[str, str]:
    if isinstance(graph, GraphStage):
        return {graph.worker_id: graph.name}
    if isinstance(graph, Sequential) or isinstance(graph, Parallel):
        res = {}
        for s in graph.sections:
            res.update(get_stage_to_worker_id(s))
        return res
    if isinstance(graph, Loop):
        return get_stage_to_worker_id(graph.section)