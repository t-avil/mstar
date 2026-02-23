

from dataclasses import dataclass, field

from mminf.graph.base import (
    GraphSection, GraphStage, SignalToDests, SignalToDestsAndFlags,
    get_signal_to_dest_mapping, get_stage_to_inputs_mapping, remove_flags
)


@dataclass
class PerRequestStageQueues:
    waiting: GraphSection | None
    ready: list[GraphStage] = field(default_factory=list)
    subgraph_id: str = field(default="")
    
    def _update_ready_waiting(self):
        """
        Moves sections from the waiting section to the ready queue,
        replaces self.waiting with whatever's left
        """
        if self.waiting is None:
            return
        new_ready, new_waiting = self.waiting.split_off_ready()
        self.ready += new_ready
        self.waiting = new_waiting
    
    def process_new_inputs(
        self,
        new_inputs: SignalToDests 
    ) -> SignalToDests:
        """
        Processes all outputs that feed into the waiting graph section, and
        return a dictionary of external output pointers
        """
        if self.waiting is None:
            return remove_flags(new_inputs)

        new_inputs = get_stage_to_inputs_mapping(remove_flags(new_inputs))
        self.waiting.ingest_inputs(new_inputs)
        external_outputs = new_inputs
        
        self._update_ready_waiting()
        return get_signal_to_dest_mapping(external_outputs)

