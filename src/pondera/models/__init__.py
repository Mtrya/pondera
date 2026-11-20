"""Neural model definitions."""

from pondera.models.board_context import BoardContextModel, SymbolicBoardModel
from pondera.models.cyfa import CyclicFlowModel
from pondera.models.delta import GatedDeltaModel, KimiDeltaModel
from pondera.models.gdn2 import GatedDeltaNet2Model
from pondera.models.kalman import KalmanDeltaModel
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
        GatedDeltaNet2Model,
        CyclicFlowModel,
        KalmanDeltaModel,
        BoardContextModel,
        SymbolicBoardModel,
    )
}
