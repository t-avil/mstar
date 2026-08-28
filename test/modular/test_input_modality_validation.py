"""Intake rejects an input modality the loaded model cannot encode.

The server-wide modality set says what M* knows about, not what the model in
front of it implements — so BAGEL, which has no audio encoder, used to accept
an audio request and fail somewhere downstream instead of at the door.
"""
import pytest

from mstar.api_server.entrypoint import APIServer
from mstar.model.base import Model


class _Model:
    SUPPORTED_INPUT_MODALITIES = frozenset({"text", "image"})


def _server(model):
    api = APIServer.__new__(APIServer)
    api.model = model
    api.model_name = "test-model"
    return api


def _submit(api, modalities):
    return APIServer.submit_request(
        api, text="hi", file_paths=None,
        input_modalities=modalities, output_modalities=["text"],
    )


@pytest.mark.parametrize("modality", ["audio", "video"])
def test_a_modality_the_model_cannot_encode_is_refused(modality):
    api = _server(_Model())
    with pytest.raises(ValueError, match="does not accept"):
        _submit(api, [modality, "text"])


def test_the_message_names_what_the_model_does_accept():
    api = _server(_Model())
    with pytest.raises(ValueError, match="image, text"):
        _submit(api, ["audio"])


def test_a_model_that_declares_nothing_keeps_the_server_wide_set():
    """Undeclared models behave exactly as they did before."""
    class _Undeclared:
        pass

    api = _server(_Undeclared())
    with pytest.raises(ValueError, match="Unsupported modality"):
        _submit(api, ["telepathy"])


def test_an_unknown_modality_is_still_refused_first():
    api = _server(_Model())
    with pytest.raises(ValueError, match="Unsupported modality"):
        _submit(api, ["telepathy"])


def test_the_base_model_declares_no_restriction():
    assert Model.SUPPORTED_INPUT_MODALITIES is None


def test_bagel_accepts_only_text_and_images():
    from mstar.model.bagel.bagel_model import BagelModel

    assert BagelModel.SUPPORTED_INPUT_MODALITIES == frozenset({"text", "image"})
