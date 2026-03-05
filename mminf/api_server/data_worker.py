

import queue
import threading
import time
from dataclasses import asdict, dataclass

import torch
import torchaudio
import torchvision
from torchcodec.decoders import VideoDecoder

from mminf.communication.communicator import CommProtocol, ZMQCommunicator
from mminf.communication.tensors import MooncakeCommunicationManager
from mminf.ipc_formats import ConductorMessage, ConductorMessageType, NewRequestConductor


@dataclass
class PreprocessInput:
    request_id: str
    text: str | None

    # file_paths is modality: list of filenames
    file_paths: dict[str, list[str]] | None
    input_modalities: list[str]
    output_modalities: list[str]
    model_kwargs: dict


def _preprocess_loop(**kwargs):
    worker = PreprocessWorkerThread(**kwargs)
    worker.run()


class PreprocessWorker:
    def __init__(
        self,
        hostname: str = "localhost",
        socket_path_prefix: str = "/tmp/mminf",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
    ):
        self.request_input_queue = queue.Queue()
        self.request_output_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.thread = threading.Thread(
            target=_preprocess_loop,
            kwargs=dict(
                in_queue=self.request_input_queue,
                out_queue=self.request_output_queue,
                stop_event=self.stop_event,
                hostname=hostname,
                socket_path_prefix=socket_path_prefix,
                tensor_comm_protocol=tensor_comm_protocol,
            )
        )

    def new_request(self, input: PreprocessInput):
        self.request_input_queue.put(input)

    def shutdown(self):
        self.stop_event.set()
        self.thread.join()


class PreprocessWorkerThread:
    def __init__(
        self,
        in_queue: queue.Queue, # for preprocessing
        out_queue: queue.Queue, # for output streaming
        stop_event: threading.Event,
        hostname: str = "localhost",
        socket_path_prefix: str = "/tmp/mminf",
        tensor_comm_protocol: CommProtocol = CommProtocol.RDMA,
        device: str = "cpu",
    ):
        self.in_queue = in_queue
        self.out_queue = out_queue # unused at the moment
        self.stop_event = stop_event
        self.device = device

        self.communicator = ZMQCommunicator(
            my_id="api_server_preprocess_worker",
            push_ids=["conductor"],
            ipc_socket_path_prefix=socket_path_prefix,
        ) # only used to send

        self.tensor_manager = MooncakeCommunicationManager(
            my_entity_id="api_server_preprocess_worker",
            hostname=hostname,
            communicator=self.communicator,
            protocol=tensor_comm_protocol,
        )


    def process_input(
        self, input: PreprocessInput
    ):
        tensors = {}
        input_metadata = {}
        # now everything is a list of tensors, even if there's only a single entry
        if input.text is not None:
            # Encode as UTF-8 bytes -> uint8 tensor
            byte_data = input.text.encode("utf-8")
            tensors["text_input"] = [torch.tensor(
                list(byte_data), dtype=torch.uint8, device=self.device
            )]

        if input.file_paths is not None:
            for modality in input.file_paths:
                key = f"{modality}_input"
                tensors[key] = []
                # TODO: maybe make a class of tensors_and_metadata later (figure out how to use metadata)
                input_metadata[key] = []

                for filepath in input.file_paths[modality]:
                    # ---- Image ----
                    if modality == "image":
                        img = torchvision.io.decode_image(filepath).to(self.device)  # uint8 CxHxW
                        img = img.float() / 255.0
                        tensors[key].append(img)
                        input_metadata[key].append({}) # cleanest, no metadata

                    # ---- Audio ----
                    elif modality == "audio":
                        waveform, sample_rate = torchaudio.load_with_torchcodec(
                            filepath,
                            channels_first=True
                        )
                        # waveform: (channels, time)
                        tensors[key].append(waveform)
                        input_metadata[key].append(dict(
                            sample_rate=sample_rate,
                            channels_first=True
                        ))

                    # ---- Video ----
                    elif modality == "video":
                        decoder = VideoDecoder(filepath, device=self.device)
                        video = torch.stack([frame for frame in decoder]).float() / 255.0
                        tensors[key].append(video)
                        input_metadata[key].append(asdict(decoder.metadata))

        initial_signals = self.tensor_manager.store_and_return_tensor_info(
            request_id=input.request_id,
            tensors=tensors # dict(modality_input: list[tensors])
        )
        for name, tensor_infos in initial_signals.items():
            self.tensor_manager.register_for_send(
                request_id=input.request_id, name=name,
                uuids=[info.uuid for info in tensor_infos]
            )

        msg = ConductorMessage(
            message_type=ConductorMessageType.NEW_REQUEST,
            body=NewRequestConductor(
                request_id=input.request_id,
                initial_signals=initial_signals,
                initial_input_modalities=input.input_modalities,
                initial_output_modalities=input.output_modalities,
                input_metadata=input_metadata,
                model_kwargs=input.model_kwargs
            ),
        )
        self.communicator.send("conductor", msg)

    def run(self):
        while not self.stop_event.is_set():
            if not self.in_queue.empty():
                self.process_input(self.in_queue.get())
            time.sleep(0.001)
