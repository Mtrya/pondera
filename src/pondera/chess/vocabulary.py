"""The frozen unified vocabulary: layout constants, loader, and regenerator.

`vocabulary.txt` next to this module holds one canonical name per token
id; the line number is the id. The layout is arithmetic: 768 occupied-square
atoms (square * 12 + piece over a1..h8 and PNBRQKpnbrqk), then self-contained
metadata atoms (side, castling, en passant, halfmove clock, repetition count),
then the move block, 20 value bins, mate-distance tokens (win 0-28, loss 0-30),
and the eight trace delimiters PAD, BACK, RESET, BOARD, FINAL, REFUTED, DEPTH,
END.

The file is the contract. This module validates the arithmetic layout against
the file at import; regenerate the file only when the vocabulary deliberately
changes (`uv run python -m pondera.chess.vocabulary`) and review the diff.
"""

import hashlib
from importlib.resources import files

# The version pins the serialized board-token format recorded in data
# manifests and checkpoints; sequence trace grammars version themselves
# separately in dataset manifests.
CHESS_TOKENIZER_VERSION = 3

# --- Layout cardinalities --- #
BOARD_PIECES = 12
BOARD_TOKENS = 64 * BOARD_PIECES
# One encoded position has at most 32 piece atoms and 8 metadata atoms.
SIDES = 2
CASTLING_RIGHTS = 8  # K Q k q, present and absent
EN_PASSANT_SQUARES = 65  # 64 squares plus no-square
HALFMOVE_CLOCKS = 101  # 0..99 and >=100
REPETITION_COUNTS = 5  # 1, 2, 3, 4, >=5
MOVES = 1969  # geometrically possible moves plus the draw claim
VALUE_BINS = 20
MATE_WINS = 29  # distance 0..28; 0 is the terminal position
MATE_LOSSES = 31  # distance 0..30
DELIMITERS = 8

# --- Segment offsets --- #
SIDE = BOARD_TOKENS
CASTLING = SIDE + SIDES
EN_PASSANT = CASTLING + CASTLING_RIGHTS
HALFMOVE = EN_PASSANT + EN_PASSANT_SQUARES
REPETITION = HALFMOVE + HALFMOVE_CLOCKS
META_END = REPETITION + REPETITION_COUNTS - 1
MOVE = META_END + 1
# Board + metadata segment size: the input-table width of board policies.
BOARD_VOCAB_SIZE = MOVE
VALUE = MOVE + MOVES
MATE_WIN = VALUE + VALUE_BINS
MATE_LOSS = MATE_WIN + MATE_WINS
DELIM = MATE_LOSS + MATE_LOSSES
PAD, BACK, RESET, BOARD, FINAL, REFUTED, DEPTH, END = (
    DELIM,
    DELIM + 1,
    DELIM + 2,
    DELIM + 3,
    DELIM + 4,
    DELIM + 5,
    DELIM + 6,
    DELIM + 7,
)
CHESS_VOCAB_SIZE = DELIM + DELIMITERS

_PATH = files("pondera.chess") / "vocabulary.txt"
if _PATH.is_file():
    LINES = tuple(_PATH.read_text(encoding="utf-8").split())
else:  # first-generation bootstrap; run this module as __main__ to create it
    LINES = ()


def _segments(lines):
    """Segment views of the vocabulary, bounded by their name prefixes."""
    board = [line for line in lines if line.startswith("board:")]
    meta = [line for line in lines if line.startswith("meta:")]
    value = [line for line in lines if line.startswith("value:")]
    mate = [line for line in lines if line.startswith("mate:")]
    delim = list(lines[-DELIMITERS:])
    moves = list(lines[len(board) + len(meta) : -len(value) - len(mate) - DELIMITERS])
    return {
        "board": board,
        "meta": meta,
        "moves": moves,
        "value": value,
        "mate": mate,
        "delim": delim,
    }


SEGMENTS = _segments(LINES)

IDX_TO_UCI_MOVE = dict(enumerate(SEGMENTS["moves"]))
UCI_MOVE_TO_IDX = {move: index for index, move in IDX_TO_UCI_MOVE.items()}


def _validate():
    """The frozen file must agree with the arithmetic layout."""
    if not LINES:
        return
    if len(LINES) != CHESS_VOCAB_SIZE:
        raise ValueError("Vocabulary file length disagrees with the layout")
    sizes = {name: len(segment) for name, segment in SEGMENTS.items()}
    expected = {
        "board": BOARD_TOKENS,
        "meta": META_END - BOARD_TOKENS + 1,
        "moves": MOVES,
        "value": VALUE_BINS,
        "mate": MATE_WINS + MATE_LOSSES,
        "delim": DELIMITERS,
    }
    for name, size in expected.items():
        if sizes[name] != size:
            raise ValueError(f"Vocabulary {name} segment disagrees with the layout")
    for token, name in [
        (0, "board:a1:P"),
        (SIDE, "meta:side:w"),
        (MOVE, "<claim_draw>"),
        (PAD, "<pad>"),
        (BACK, "<back>"),
        (RESET, "<reset>"),
        (BOARD, "<board>"),
        (FINAL, "<final>"),
        (REFUTED, "<refuted>"),
        (DEPTH, "<depth>"),
        (END, "<end>"),
    ]:
        if LINES[token] != name:
            raise ValueError(f"Vocabulary id {token} is not {name}")


_validate()


def _square(index):
    return f"{chr(ord('a') + index % 8)}{index // 8 + 1}"


def _move_vocabulary():
    """Every geometrically possible move plus the draw claim, sorted."""
    moves = set()
    for origin in range(64):
        origin_rank, origin_file = divmod(origin, 8)
        for target in range(64):
            if origin == target:
                continue
            target_rank, target_file = divmod(target, 8)
            dr, df = target_rank - origin_rank, target_file - origin_file
            # King, knight, rook, or bishop geometry; queen and pawn moves are
            # covered by these four cases.
            if (
                max(abs(dr), abs(df)) == 1
                or (abs(dr), abs(df)) in ((1, 2), (2, 1))
                or dr == 0
                or df == 0
                or abs(dr) == abs(df)
            ):
                moves.add(_square(origin) + _square(target))
    for file in range(8):
        # White promotes rank 7 -> 8, black rank 2 -> 1, by push or capture.
        for origin_rank, target_rank in ((6, 7), (1, 0)):
            for df in (-1, 0, 1):
                if 0 <= file + df < 8:
                    base = _square(origin_rank * 8 + file) + _square(
                        target_rank * 8 + file + df
                    )
                    moves.update(base + piece for piece in "qrbn")
    moves.add("<claim_draw>")
    return sorted(moves)


def chess_vocabulary():
    """The canonical name of every token id, in id order."""
    lines = [
        f"board:{_square(square)}:{piece}"
        for square in range(64)
        for piece in "PNBRQKpnbrqk"
    ]
    lines += [f"meta:side:{side}" for side in "wb"]
    lines += [
        f"meta:castling:{right}{'+' if flag else '-'}"
        for right in "KQkq"
        for flag in (0, 1)
    ]
    lines += [f"meta:ep:{_square(square)}" for square in range(64)]
    lines += ["meta:ep:-"]
    lines += [
        f"meta:halfmove:{count}{'+' if count == HALFMOVE_CLOCKS - 1 else ''}"
        for count in range(HALFMOVE_CLOCKS)
    ]
    lines += [f"meta:repetition:{count}" for count in range(1, REPETITION_COUNTS)]
    lines += [f"meta:repetition:{REPETITION_COUNTS}+"]
    lines += _move_vocabulary()
    lines += [f"value:bin:{bin:02d}" for bin in range(VALUE_BINS)]
    lines += [f"mate:win:{distance}" for distance in range(MATE_WINS)]
    lines += [f"mate:loss:{distance}" for distance in range(MATE_LOSSES)]
    lines += [
        "<pad>",
        "<back>",
        "<reset>",
        "<board>",
        "<final>",
        "<refuted>",
        "<depth>",
        "<end>",
    ]
    return lines


def main():
    lines = chess_vocabulary()
    old = list(LINES)
    if old == lines:
        print(f"unchanged: {len(lines)} tokens")
        return
    payload = "\n".join(lines) + "\n"
    with _PATH.open("w", encoding="utf-8") as handle:
        handle.write(payload)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    changed = "initialized" if not old else "rewritten"
    print(f"{changed}: {len(lines)} tokens, sha256 {digest}")


if __name__ == "__main__":
    main()
