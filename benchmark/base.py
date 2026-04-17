from abc import ABC, abstractmethod
from enum import Enum


class Status(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PROGRESS = "progress"


class RequestType(Enum):
    T2T = "text_to_text"
    T2I = "text_to_image"
    I2T = "image_to_text"
    I2I = "image_to_image"

    def get_output_modalities(self):
        if self in [RequestType.I2I, RequestType.T2I]:
            return "image"
        return "text"


class Model(ABC):
    def __init__(self, **kwargs):
        self.config = kwargs
    
    def get_model_kwargs(self, request_type: RequestType):
        return {}

    @abstractmethod
    def get_hf_url(self):
        pass


class Bagel(Model):
    def __init__(self, disable_cfg: bool=False, **kwargs):
        self.disable_cfg = disable_cfg

    def get_model_kwargs(self, request_type: RequestType):
        if self.disable_cfg:
            return {
                "cfg_img_scale": 1.0,
                "cfg_text_scale": 1.0,
            }
        if request_type == RequestType.I2I:
            return {
                "cfg_img_scale": 2.0,
                "cfg_interval": [0.0, 1.0],
                "cfg_renorm_type": "text_channel",
            }
        return {}

    def get_hf_url(self):
        return "ByteDance-Seed/BAGEL-7B-MoT"


class ModelType(Enum):
    BAGEL = "bagel"

    def inst(self, **kwargs) -> Model:
        if self == ModelType.BAGEL:
            return Bagel(**kwargs)
        raise NotImplementedError(f"Unknown model type {self}")