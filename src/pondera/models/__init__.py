"""Neural model definitions."""

from pondera.models.board_context import BoardContextModel, SymbolicBoardModel
from pondera.models.delta import GatedDeltaModel, KimiDeltaModel
from pondera.models.linear import LinearVanillaModel
from pondera.models.looped import LoopedModel, RecurrentDepthModel
from pondera.models.vanilla import VanillaModel

MODEL_CLASSES = {
    model.architecture: model
    for model in (
        VanillaModel,
        LinearVanillaModel,
        LoopedModel,
        RecurrentDepthModel,
        GatedDeltaModel,
        KimiDeltaModel,
        BoardContextModel,
        SymbolicBoardModel,
    )
}
