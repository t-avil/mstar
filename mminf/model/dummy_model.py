from copy import deepcopy

from mminf.graph.base import GraphPointer, GraphStage, Loop, Parallel, Sequential
from mminf.graph.worker_assignment import Subgraph
from mminf.model.base import STREAM_OUT, Model, TensorData


class DummyModel(Model):
    """
    Show-o2-inspired model that does nothing, for testing and example purposes.
    """
    def _get_text_emb(self):
        return GraphStage(
            name="text_emb",
            input_ids=["text"],
            outputs={
                "text_emb": "LLM"
            }
        )
    
    def _get_img_emb(self):
        return GraphStage(
            name="image_emb",
            input_ids=["images"],
            outputs={
                "img_emb": "LLM"
            }
        )

    def get_phases(self):
        prefill = Sequential([
            Parallel([self._get_text_emb(), self._get_img_emb()]),
            GraphStage(
                name="LLM",
                input_ids=["text_emb", "img_emb"],
                outputs={
                    "new_token": GraphPointer(STREAM_OUT, back_to_conductor=True)
                }
            )
        ])
        decode = deepcopy(prefill)
        image_gen = Sequential([
            Parallel([self._get_text_emb(), self._get_img_emb()]),
            Loop(
                section=Sequential([
                    GraphStage(
                        name="LLM",
                        input_ids=["text_emb", "img_emb", "latents"],
                        outputs={
                            "hidden_states": GraphPointer("flow")
                        }
                    ),
                    GraphStage(
                        "flow",
                        input_ids=["hidden_states"],
                        outputs={
                            "latents": GraphPointer("LLM")
                        }
                    )
                ]),
                n_iters=10,
                outputs={
                    "latents": GraphPointer("VAE_dec")
                }
            ),
            GraphStage(
                name="VAE_dec",
                input_ids=["latents"],
                outputs={
                    "image_output": GraphPointer(STREAM_OUT)
                }
            )
        ])

        return dict(
            prefill=prefill,
            decode=decode,
            image_gen=image_gen
        )


    def get_active_subgraph_ids(
        self, input_modalities: list[str],
        output_modalities: list[str],
        **kwargs
    ) -> list[str]:
        pass

    def step(
        self, stage_name: str,
        input_tensors: dict[str, TensorData],
        state, # TODO: figure out state
        **kwargs
    ):
        return # do nothing
