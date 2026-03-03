from abc import ABC, abstractmethod
from dataclasses import dataclass
try:
    from mooncake.engine import TransferEngine
except ImportError:
    TransferEngine = None
import torch


from mminf.communication.communicator import BaseCommunicator, CommProtocol
from mminf.graph.base import GraphPointer, TensorPointerInfo
from mminf.ipc_formats import NameAndAddress, TensorReceived, WorkerMessage, WorkerMessageType


@dataclass(frozen=True)
class NameAndRequestId:
    tensor_name: str
    request_id: str


@dataclass
class GraphPtrAndLocalAddr:
    graph_pointer: GraphPointer
    local_address: int


@dataclass
class EventAndPointers:
    event: torch.cuda.Event
    pointers: list[GraphPtrAndLocalAddr]
    request_id: str = ""


class TensorCommunicationManager(ABC):
    @abstractmethod
    def register_and_return_tensor_info(
        self, request_id: str, tensors: dict[str, list[torch.Tensor]],
    ) -> dict[str, list[TensorPointerInfo]]:
        """
        If relevant (e.g., mooncake rdma), registers buffers.
        Returns tensor name to TensorPointerInfo (contains addresses, datatypes,
        num bytes, etc.) for each tensor.
        """
        pass

    @abstractmethod
    def register_and_populate_graph_edges(
        self, request_id: str, tensors: dict[str, list[torch.Tensor]],
        graph_pointers: list[GraphPointer]
    ):
        """
        Updates graph_pointers with required tensor info (addresses, datatypes,
        num bytes, etc.).
        If relevant (e.g., mooncake rdma), registers buffers.
        """
        pass

    @abstractmethod
    def get_tensor(self, request_id: str, tensor_name: str, local_addr: int=None) -> torch.Tensor:
        pass

    @abstractmethod
    def cleanup(self, request_id: str, tensor_name: str, address: int | None=None):
        """
        Removes buffer if exists. Unregisters buffers if relevant
        """
        pass

    @abstractmethod
    def cleanup_request(self, request_id: str):
        """
        Removes all tensors for a given request. Unregisters buffers if relevant.
        """
        pass

    @abstractmethod
    def start_read_tensors(
        self, request_id: str, graph_pointers: list[GraphPointer]
    ) -> list[int]:
        """
        Initializes empty buffer, initializes a read. May return immediately.
        """
        pass

    @abstractmethod
    def get_ready_tensors(self) -> dict[str, list[GraphPtrAndLocalAddr]]:
        """
        Returns request_id: list of the GraphPointers that are currently
        ready for that request

        Returns a list of local addresses for the tensors being read.
        """
        pass


class MooncakeCommunicationManager(TensorCommunicationManager):
    def __init__(
        self,
        my_entity_id: str,
        hostname: str,
        communicator: BaseCommunicator,
        protocol: CommProtocol=CommProtocol.RDMA,
        metadata_server: str="P2PHANDSHAKE" # [ETCD_SERVER_URL, P2PHANDSHAKE, ...]
        
    ):
        self.my_entity_id = my_entity_id

        # NOTE: none of the designs for how to handle the values of self.tensors
        # being a list are set in stone. This is a "reasonable initial
        # implementation," but should change based on what makes sense for
        # downstream changes (right now the code is "leaving breadcrumbs
        # for a hypothetical thinker-talker", but we will have a better idea of
        # what this should look like while implementing the actual thinker-
        # talker relay)

        # We store a mapping of tensor (local) address to tensors. We use this
        # instead of a queue because we aren't sure if the FIFO property will
        # hold in all cases. We use the dict instead of a list because the index
        # of a list would change whenever tensors are removed from the list.
        # We can also, e.g., give each tensor a UUID.
        self.tensors: dict[NameAndRequestId, dict[int, torch.Tensor]] = {}

        self.communicator = communicator
        self.protocol = protocol
        self.my_session_id = communicator.get_session_id()

        if TransferEngine is not None:
            self.engine = TransferEngine()
            self.engine.initialize(
                hostname,
                metadata_server,
                protocol,
                ""
            )
        else:
            self.engine = None
        # Per-transfer pending list (allows partial tensor readiness)
        self.pending: list[EventAndPointers] = []

    def register_and_return_tensor_info(
        self, request_id: str, tensors: dict[str, list[torch.Tensor]],
    ) -> dict[str, list[TensorPointerInfo]]:
        tensor_info: dict[str, list[TensorPointerInfo]] = {}
        for name in tensors:
            key = NameAndRequestId(
                name, request_id=request_id
            )
            tensor_info[name] = []
            self.tensors[key] = self.tensors.get(key, {})

            for tensor in tensors[name]:
                self.tensors[key][tensor.data_ptr()] = tensor
                new_tensor_info = TensorPointerInfo(
                    dims=tensor.shape,
                    dtype=tensor.dtype,
                    nbytes=tensor.element_size() * tensor.nelement(),
                    address=tensor.data_ptr(),
                    source_session_id=self.my_session_id,
                    source_entity=self.my_entity_id
                )
                tensor_info[name].append(new_tensor_info)
            
                if self.protocol == CommProtocol.RDMA:
                    ret_value = self.engine.register_memory(
                        new_tensor_info.address, new_tensor_info.nbytes
                    )
                    if ret_value != 0:
                        # TODO: error handling
                        raise RuntimeError("Mooncake memory registration failed.")
        return tensor_info
        
    def register_and_populate_graph_edges(
        self, request_id: str, tensors: dict[str, list[torch.Tensor]],
        graph_pointers: list[GraphPointer]
    ):
        # get tensor name to graph pointers
        name_to_pointers: dict[str, list[GraphPointer]] = {}
        for pointer in graph_pointers:
            if pointer.name not in name_to_pointers:
                name_to_pointers[pointer.name] = []
            name_to_pointers[pointer.name].append(pointer)

        pointer_info = self.register_and_return_tensor_info(
            request_id=request_id, tensors=tensors
        )
        for name in tensors:
            for pointer, pointer_info_elem in zip(name_to_pointers[name], pointer_info[name]):
                pointer.tensor_info = pointer_info_elem
            
    def cleanup(self, request_id: str, tensor_name: str, addresses: list[int] | None=None):
        key = NameAndRequestId(tensor_name, request_id)
        if key not in self.tensors:
            return
        
        # By default, cleanup all tensors with the given key, unless the address
        # argument is provided
        if addresses is None:
            addresses = list(self.tensors[key].keys())

        for addr in addresses:
            if addr not in self.tensors[key]:
                continue
            if self.protocol == CommProtocol.RDMA and self.engine is not None:
                ret_value = self.engine.unregister_memory(
                    self.tensors[key][addr].data_ptr()
                )
                if ret_value != 0:
                    raise RuntimeError("Mooncake memory unregistration failed.")
            del self.tensors[key][addr]

    def cleanup_request(self, request_id: str):
        keys_to_remove = [
            key for key in self.tensors if key.request_id == request_id
        ]
        for key in keys_to_remove:
            for tensor in self.tensors[key].values():
                if self.protocol == CommProtocol.RDMA and self.engine is not None:
                    self.engine.unregister_memory(tensor.data_ptr())
        self.tensors[key].clear()
        # Also remove any pending transfers for this request
        self.pending = [
            ep for ep in self.pending if ep.request_id != request_id
        ]

    def get_tensor(self, request_id: str, tensor_name: str, local_addr: int=None) -> torch.Tensor:
        # It is the burden of the worker to access the correct local_addr.
        # For now (in our default non-thinker-talker request flow), we assume
        #  that the element of self.tensors is a singleton; this is ensured by
        # worker.py -> _cleanup_consumed_inputs.

        if local_addr is None:
            local_addr = list(self.tensors[NameAndRequestId(
                tensor_name, request_id
            )].keys())[0] # in modern Python, this is the oldest element of the dict
        return self.tensors[NameAndRequestId(
            tensor_name, request_id
        )][local_addr]

    def get_ready_tensors(self) -> dict[str, list[GraphPtrAndLocalAddr]]:
        """
        Poll CUDA events. Return {request_id: [ready GraphPointers]}.
        Remove completed entries from self.pending.
        Sends TENSOR_RECEIVED ACKs back to senders so they can free buffers.
        """
        # For now, it is the burden of the receiver, in the case they are
        # reading from a stream/relay, to correctly map the local address
        # in the returned GraphPtrAndLocalAddr objects to the right index
        # of get_tensor. This can theoretically be done via a combination of
        # the list of addresses returned by start_read_tensors(...) and proper
        # cleanup of tensors after they are used / batching of reads.
        ready: dict[str, list[GraphPtrAndLocalAddr]] = {}
        still_pending = []
        # Collect ACKs to send: (source_entity, request_id) -> tensor_names and
        # **remote (not local) address**
        acks: dict[tuple[str, str], list[NameAndAddress]] = {}

        for ep in self.pending:
            if ep.event.query():
                for ptr in ep.pointers:
                    ready.setdefault(ep.request_id, []).append(ptr)
                    if ptr.graph_pointer.tensor_info is not None:
                        key = (ptr.graph_pointer.tensor_info.source_entity, ep.request_id)
                        acks.setdefault(key, []).append(NameAndAddress(
                            tensor_id=ptr.graph_pointer.name,
                            address=ptr.graph_pointer.tensor_info.address
                        ))
            else:
                still_pending.append(ep)
        self.pending = still_pending

        # Send ACKs to senders
        for (source_entity, request_id), tensor_name_addrs in acks.items():
            if source_entity == self.my_entity_id:
                continue  # local transfer, no ACK needed
            self.communicator.send(
                source_entity,
                WorkerMessage(
                    message_type=WorkerMessageType.TENSOR_RECEIVED,
                    body=TensorReceived(
                        request_id=request_id,
                        successful_tensors=tensor_name_addrs,
                        failed_tensor_ids=[],
                    ),
                ),
            )

        return ready

    def start_read_tensors(
        self, request_id: str, graph_pointers: list[GraphPointer]
    ) -> list[int]:
        """
        For each pointer with tensor_info (RDMA source): allocate dst tensor,
        register memory, call engine.transfer_read_on_cuda(), record CUDA event.
        For each pointer WITHOUT tensor_info (signal-only): no data to transfer.

        Returns a list of local addresses for the tensors being read.
        """
        stream = torch.cuda.Stream()
        addrs = []
        for ptr in graph_pointers:
            if ptr.tensor_info is None:
                continue  # signal-only pointer, no data to transfer

            info = ptr.tensor_info
            dst = torch.empty(info.dims, dtype=info.dtype, device="cuda")
            addrs.append(dst.data_ptr())
            self.tensors[NameAndRequestId(
                ptr.name, request_id
            )][dst.data_ptr()] = dst

            if self.protocol == CommProtocol.RDMA:
                self.engine.register_memory(dst.data_ptr(), info.nbytes)

                with torch.cuda.stream(stream):
                    self.engine.transfer_read_on_cuda(
                        info.source_session_id,
                        dst.data_ptr(),
                        info.address,
                        info.nbytes,
                        stream.cuda_stream,
                    )
                event = torch.cuda.Event()
                event.record(stream)
                self.pending.append(
                    EventAndPointers(
                        event=event, pointers=[GraphPtrAndLocalAddr(
                            graph_pointer=ptr,
                            local_address=dst.data_ptr()
                        )], request_id=request_id
                    )
                )
        return addrs