"""Absolute input offsets of prior active-branch entries at navigation tokens."""

import numpy as np

from pondera.chess.merge import MERGE_ID_BASE
from pondera.chess.vocabulary import (
    BACK,
    BOARD,
    FINAL,
    IDX_TO_UCI_MOVE,
    META_END,
    MOVE,
    REPETITION,
    RESET,
)

ENTER = MOVE
ROOT = REPETITION
MOVES = frozenset(
    MOVE + i for i, uci in IDX_TO_UCI_MOVE.items() if uci != "<claim_draw>"
)


def navigation_table(dictionary):
    """Resolve composite navigation in atom order once per frozen vocabulary."""
    atoms = tuple((i,) for i in range(MERGE_ID_BASE))
    operations = []
    for expansion in (*atoms, *dictionary.entries):
        ops = []
        for atom in expansion:
            if atom in MOVES:
                ops.append(ENTER)
            elif REPETITION <= atom <= META_END:
                ops.append(ROOT)
            elif atom in (BACK, BOARD, RESET, FINAL):
                ops.append(atom)
        operations.append(tuple(ops))
    return tuple(operations)


class BranchEntries:
    def __init__(self):
        self.root = -1
        self.path = []
        self.latest_move = -1
        self.in_board = True
        self.final = False

    def consume(self, index, operations):
        """Return matched branch and chronological sources at a visible BACK or RESET."""
        navigation = False
        for op in operations:
            if op == FINAL:
                self.final = True
                break
            if self.final:
                break
            if op == ROOT:
                if self.root < 0:
                    self.root = index
                self.in_board = False
            elif self.in_board:
                continue
            elif op == BOARD:
                self.in_board = True
            elif op == ENTER:
                self.path.append(index)
                self.latest_move = index
            elif op == RESET:
                self.path.clear()
                navigation = True
            elif self.path:
                self.path.pop()
                navigation = True
        source = self.path[-1] if self.path else self.root
        latest = self.latest_move if self.latest_move >= 0 else self.root
        if (
            navigation
            and not self.final
            and 0 <= source < index
            and 0 <= latest < index
        ):
            return source, latest
        return -1, -1


def navigation_sources(tokens, boundaries, table):
    sources = np.full((len(tokens), 2), -1, dtype=np.int32)
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        cursor = BranchEntries()
        for index in range(start, stop):
            ops = table[int(tokens[index])]
            if ops:
                sources[index] = cursor.consume(index, ops)
    return sources
