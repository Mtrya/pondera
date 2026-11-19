"""A causal transformer with weight sharing across depth in the middle blocks."""

import random
from itertools import chain, repeat

from pondera.models.vanilla import VanillaModel


class LoopedModel(VanillaModel):
    architecture = "looped"

    def __init__(self, *, loop_repetitions=3, **kwargs):
        if kwargs["num_blocks"] < 5:
            raise ValueError("A looped model needs two blocks on each side of its core")
        if loop_repetitions < 1:
            raise ValueError("Require at least one repetition")
        super().__init__(**kwargs)
        self.loop_repetitions = loop_repetitions
        self.training_repetitions = loop_repetitions

    def select_training_depth(self, seed, update):
        self.training_repetitions = random.Random(f"{seed}:loop:{update}").randint(
            1, self.loop_repetitions
        )
        return self.training_repetitions

    def execution_blocks(self):
        repetitions = (
            self.training_repetitions if self.training else self.loop_repetitions
        )
        core = self.blocks[2:-2]
        return chain(
            self.blocks[:2],
            chain.from_iterable(repeat(core, repetitions)),
            self.blocks[-2:],
        )
