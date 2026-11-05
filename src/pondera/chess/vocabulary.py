"""Generate the canonical base token IDs from the arithmetic chess layout.

The order is occupied-square atoms, board metadata, sorted geometric moves,
value bins, mate distances, then trace delimiters. Changing this order changes
serialized IDs and requires a deliberate tokenizer-version change.
"""

# The version pins the serialized board-token format recorded in data
# manifests and checkpoints; trace grammars version themselves separately in
# dataset manifests.
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
MATE_WINS = 29  # distance 0..28; 28 is the open-ended 28+ bucket
MATE_LOSSES = 31  # distance 0..30; 30 is the open-ended 30+ bucket
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


LINES = tuple(chess_vocabulary())
IDX_TO_UCI_MOVE = dict(enumerate(LINES[MOVE:VALUE]))
UCI_MOVE_TO_IDX = {move: index for index, move in IDX_TO_UCI_MOVE.items()}


def main():
    """Print canonical token names in ID order."""
    print("\n".join(LINES))


if __name__ == "__main__":
    main()
