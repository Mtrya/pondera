"""Neural model definitions."""

from pondera.models.delta import GatedDeltaModel, KimiDeltaModel
from pondera.models.linear import LinearVanillaModel
from pondera.models.looped import LoopedModel
from pondera.models.vanilla import VanillaModel

MODEL_CLASSES = {
    model.architecture: model
    for model in (
        VanillaModel,
        LinearVanillaModel,
        LoopedModel,
        GatedDeltaModel,
        KimiDeltaModel,
    )
}
