"""Rule-only path state machine for trace thought traces.

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
for game rules. Two repetition readings stay separate: ``repetition`` is the
observation's count and includes the root's prior history, while
``search_terminal`` applies the engine's search rule, which knows only the
written path and ignores that history.
"""

import chess

from pondera.chess.codec import encode_board


def repetition_count(board: chess.Board) -> int:
    """Occurrences of the current position in the board's move history, capped at five."""
    for count in range(5, 1, -1):
        if board.is_repetition(count):
            return count
    return 1


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

    def search_terminal(self):
        """Search-rule terminal verdict for the current node.

        Returns ``"mate"`` for checkmate, ``"draw"`` for stalemate, repetition,
        or the 50-move rule, and ``None`` when the search may continue. Only
        the written path counts: the root's pre-history repetition is ignored,
        and a second occurrence of the root position is not a draw while a
        third is. A position that occurred once strictly after the root is a
        draw. The 50-move rule is a claim, not an automatic root terminal: the
        root is still searched so its candidates and claim can be recorded,
        while a node below the root stops there. Checkmate takes precedence
        over the 50-move rule, and insufficient material is not a terminal:
        the engine keeps searching there, so ``board.is_game_over()`` must not
        stand in for this rule.
        """
        if self.board.is_checkmate():
            return "mate"
        if self.board.is_stalemate():
            return "draw"
        current = self._position_key(self.board)
        prior = sum(
            1 for board in self._stack[:-1] if self._position_key(board) == current
        )
        if prior >= 2 or (prior == 1 and current != self._root_key):
            return "draw"
        if self._path and self.board.halfmove_clock >= 100:
            return "draw"
        return None

    def board_tokens(self):
        """Current position as piece-joint atoms plus eight metadata atoms."""
        return encode_board(self.fen(), self.repetition())
