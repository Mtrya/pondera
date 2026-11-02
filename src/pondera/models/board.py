"""Board-token orientation shared by policy inputs."""

import torch

from pondera.chess.symmetry import TOKEN_MIRROR
from pondera.chess.vocabulary import SIDE

_MIRROR = torch.from_numpy(TOKEN_MIRROR)


def black_to_move(ids):
    return (ids == SIDE + 1).any(-1)


def reflect_board_tokens(ids, reflected):
    mirrored = _MIRROR.to(ids.device)[ids.long()]
    return torch.where(reflected[:, None], mirrored, ids.long()).to(ids.dtype)
