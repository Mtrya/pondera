"""Rule-only path state machine for sequence thought traces.

A tracker owns the ground truth of one thought path: a root position with its
repetition count, a stack of absolute UCI moves, and the current python-chess
board. It never evaluates moves, never searches, and imports no data, model, or
training code.

A move token in a trace identifies a child node of the current node. Pushing a
legal move is the only navigation primitive: when the child does not exist yet
it is an expansion, when it already exists it is a location, and the rule layer
does not need to tell the two apart. BACK returns to the parent, RESET returns
to the root from any depth, and both leave evaluations untouched.

The observation of a node is the board encoding of its absolute position:
occupied-square piece-joint atoms plus the eight metadata atoms, including
saturated halfmove and repetition counters. The tracker retains exact counters
for game rules. Repetition is the number
of times the current position has occurred, counting the root's prior history:
python-chess judges repetitions along the current path, and a current position
equal to the root adds the root's pre-history occurrences.
"""

import chess

from pondera.chess.codec import encode_board


class PathTracker:
    """Root position plus an absolute-UCI path and its current board."""

    def __init__(self, root_fen: str, repetition: int = 1):
        if repetition < 1:
            raise ValueError("Root repetition must be positive")
        self.root_fen = root_fen
        self.root_repetition = repetition
        self._root = chess.Board(root_fen)
        self._root_key = self._position_key(self._root)
        self.board = self._root.copy()
        self._path = []
        self._stack = [self._root.copy()]

    @staticmethod
    def _position_key(board):
        """Position identity used by repetition rules, without move counters.

        python-chess writes the en-passant field only when an en-passant
        capture is actually legal, which is exactly when it distinguishes
        positions for repetition.
        """
        return " ".join(board.fen().split()[:4])

    @property
    def path(self) -> tuple[str, ...]:
        """Absolute UCI moves from the root to the current node."""
        return tuple(self._path)

    @property
    def depth(self) -> int:
        return len(self._path)

    def fen(self) -> str:
        """Current FEN, preserving the en-passant square as stored."""
        return self.board.fen(en_passant="fen")

    def legal_moves(self) -> list[str]:
        """Absolute UCI moves that are legal at the current node."""
        return sorted(move.uci() for move in self.board.legal_moves)

    def push(self, move: str) -> None:
        """Enter the child reached by a legal absolute UCI move."""
        try:
            parsed = chess.Move.from_uci(move)
        except ValueError as error:
            raise ValueError(f"Not a board move: {move}") from error
        if parsed not in self.board.legal_moves:
            raise ValueError(f"Illegal move at this node: {move}")
        self.board.push(parsed)
        self._path.append(move)
        self._stack.append(self.board.copy())

    def back(self) -> None:
        """Return to the parent node; illegal at the root."""
        if not self._path:
            raise ValueError("BACK is illegal at the root")
        self._path.pop()
        self._stack.pop()
        self.board = self._stack[-1].copy()

    def reset(self) -> None:
        """Return to the root node from any depth."""
        self._path.clear()
        self._stack = [self._root.copy()]
        self.board = self._root.copy()

    def repetition(self) -> int:
        """Occurrences of the current position, including the root history."""
        count = 1
        while count < len(self._stack) and self.board.is_repetition(count + 1):
            count += 1
        if self._position_key(self.board) == self._root_key:
            count += self.root_repetition - 1
        return count

    def board_tokens(self):
        """Current position as piece-joint atoms plus eight metadata atoms."""
        return encode_board(self.fen(), self.repetition())
