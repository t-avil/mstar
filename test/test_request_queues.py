import sys

from mminf.graph.request_queues import RequestQueues
sys.path.insert(0, ".")
from mminf.graph.base import GraphPointer, GraphStage, Loop, Parallel, Sequential
import numpy as np


if __name__ == "__main__":
    # show-o2-style graph with weird stuff added to stress-test

    loop = Loop(
        section=Parallel([
            Sequential([
                GraphStage(
                    name="LLM",
                    input_ids=["text_emb", "img_emb", "latents"],
                    outputs={
                        "hidden_states": [GraphPointer("flow")],
                        "some_random_external_output": [GraphPointer("f")]
                    }
                ),
                Loop(
                    section=Sequential([
                        GraphStage(
                            name="flow",
                            input_ids=["hidden_states", "mystery", "mystery2"],
                            outputs={
                                "mystery2": [GraphPointer("flow")],
                                "partial_latents": [GraphPointer("flow2")]
                            }
                        ),
                        GraphStage(
                            name="flow2",
                            input_ids=["partial_latents"],
                            outputs={
                                "latents": []
                            }
                        ),
                    ]),
                    n_iters=2,
                    outputs={
                        "latents": [GraphPointer("LLM")],
                    }
                )
            ]),
            Sequential([
                GraphStage(
                    name="f",
                    input_ids=["mystery", "some_random_external_output"],
                    outputs={
                        "xyz": [GraphPointer("g")]
                    }
                ),
                GraphStage(
                    name="g",
                    input_ids=["xyz"],
                    outputs={
                        "mystery": [GraphPointer("f"), GraphPointer("flow")]
                    }
                )
            ])
        ]),
        n_iters=3,
        outputs={
            "latents": [GraphPointer("VAE_decoder")],
            "some_random_external_output": [GraphPointer("STREAM_OUT")]
        }
    )

    loop = Parallel([
        Sequential([
            Loop(
                section=loop.section.sections[0].sections[0],
                n_iters=loop.n_iters,
                curr_iter=loop.curr_iter,
                external_inputs=loop.external_inputs,
                loop_back_signals=loop.loop_back_signals,
                outputs=loop.outputs
            ),
            Loop(
                section=loop.section.sections[0].sections[1],
                n_iters=loop.n_iters,
                curr_iter=loop.curr_iter,
                external_inputs=loop.external_inputs,
                loop_back_signals=loop.loop_back_signals,
                outputs=loop.outputs
            )
        ]),
        Loop(
            section=loop.section.sections[1],
            n_iters=loop.n_iters,
            curr_iter=loop.curr_iter,
            external_inputs=loop.external_inputs,
            loop_back_signals=loop.loop_back_signals,
            outputs=loop.outputs
        )
    ])


    network = Sequential([
        Parallel([
            GraphStage(
                name="text_emb",
                input_ids=["text"],
                outputs={
                    "text_emb": [GraphPointer("LLM")],
                    "mystery": [GraphPointer("flow"), GraphPointer("f")]
                }
            ),
            GraphStage(
                name="vit_encoder",
                input_ids=["image"],
                outputs={
                    "img_emb": [GraphPointer("LLM")],
                    "some_random_external_output": [GraphPointer("f")]
                }
            )
        ]),
        Loop(
            section=Parallel([
                Sequential([
                    GraphStage(
                        name="LLM",
                        input_ids=["text_emb", "img_emb", "latents"],
                        outputs={
                            "hidden_states": [GraphPointer("flow")],
                            "some_random_external_output": [GraphPointer("f")]
                        }
                    ),
                    Loop(
                        section=Sequential([
                            GraphStage(
                                name="flow",
                                input_ids=["hidden_states", "mystery", "mystery2"],
                                outputs={
                                    "mystery2": [GraphPointer("flow")],
                                    "partial_latents": [GraphPointer("flow2")]
                                }
                            ),
                            GraphStage(
                                name="flow2",
                                input_ids=["partial_latents"],
                                outputs={
                                    "latents": []
                                }
                            ),
                        ]),
                        n_iters=2,
                        outputs={
                            "latents": [GraphPointer("LLM")],
                        }
                    )
                ]),
                Sequential([
                    GraphStage(
                        name="f",
                        input_ids=["mystery", "some_random_external_output"],
                        outputs={
                            "xyz": [GraphPointer("g")]
                        }
                    ),
                    GraphStage(
                        name="g",
                        input_ids=["xyz"],
                        outputs={
                            "mystery": [GraphPointer("f"), GraphPointer("flow")]
                        }
                    )
                ])
            ]),
            n_iters=3,
            outputs={
                "latents": [GraphPointer("VAE_decoder")],
                "some_random_external_output": [GraphPointer("STREAM_OUT")]
            }
        ),
        GraphStage(
            name="VAE_decoder",
            input_ids=["latents"],
            outputs={
                "generated_image": [GraphPointer("STREAM_OUT")]
            }
        )
    ])

    provided_inputs = {
        "text": [GraphPointer("text_emb")],
        "image": [GraphPointer("vit_encoder")],
        "latents": [GraphPointer("LLM")],
        "mystery2": [GraphPointer("flow")]
    }

    queues = RequestQueues(
        ready=[],
        waiting=network
    )
    queues.process_new_inputs(provided_inputs)

    # loop until all stages are done and print out
    while len(queues.ready) > 0 or queues.waiting is not None:
        print("\n" + "="*60)
        print("Ready stages:", [stage.name for stage in queues.ready])
        if queues.waiting is not None:
            print("Waiting stages:", queues.waiting.get_stage_names())

        if len(queues.ready) == 0:
            print(queues.waiting)
            raise Exception("No ready stages but still waiting stages, something's wrong")
        print()
        # pop a random ready stage and process it
        stage = queues.ready.pop(np.random.randint(0, len(queues.ready)))
        print(f"Processing stage {stage.name} with inputs {stage.input_ids}")
        new_inputs = stage.outputs
        external_outputs = queues.process_new_inputs(new_inputs)
        if external_outputs:
            print(f"External outputs: {external_outputs}")
       
    