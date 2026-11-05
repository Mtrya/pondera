"""Root records, deterministic splits, stratified sampling, and pool storage."""

import heapq
import json
import random
from collections import Counter
from pathlib import Path

import chess

from pondera.data.dataset import write_manifest

SPLIT_MODULUS = 65
VAL_REMAINDER = 0
FIVEFOLD = 5
CLAIM_HALFMOVE = 100


class CorpusError(RuntimeError):
    """A root pool that cannot be reproduced from its inputs."""


def validate_record(record, index=0):
    root_id = record.get("root_id")
    split = record.get("split")
    fen = record.get("fen")
    repetition = record.get("repetition")
    if not isinstance(root_id, str) or not root_id:
        raise CorpusError(f"Root on line {index + 1} has no root_id")
    if split not in {"train", "val"}:
        raise CorpusError(f"Root {root_id} has an unknown split")
    if not isinstance(fen, str) or len(fen.split()) != 6:
        raise CorpusError(f"Root {root_id} has an invalid FEN")
    if (
        not isinstance(repetition, int)
        or isinstance(repetition, bool)
        or not 1 <= repetition < FIVEFOLD
    ):
        raise CorpusError(f"Root {root_id} has an invalid repetition count")
    position_key = _position_key(fen)
    if record.get("position_key", position_key) != position_key:
        raise CorpusError(f"Root {root_id} has an inconsistent position key")
    try:
        board = chess.Board(fen)
    except ValueError as error:
        raise CorpusError(f"Root {root_id} has an invalid FEN") from error
    if not board.is_valid():
        raise CorpusError(f"Root {root_id} has an invalid position")
    if board.is_checkmate() or board.is_stalemate() or not board.legal_moves:
        raise CorpusError(f"Root {root_id} has no legal move")
    turn = "white" if board.turn == chess.WHITE else "black"
    if record.get("turn") not in (None, turn):
        raise CorpusError(f"Root {root_id} has a turn inconsistent with its FEN")
    return {**record, "turn": turn, "position_key": position_key}


def read_roots(path):
    roots = []
    seen = set()
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise CorpusError(f"Invalid root JSON on line {index + 1}") from error
            if not isinstance(record, dict):
                raise CorpusError(f"Root record on line {index + 1} is not an object")
            root = validate_record(record, index)
            if root["root_id"] in seen:
                raise CorpusError(f"Duplicate root id: {root['root_id']}")
            seen.add(root["root_id"])
            roots.append(root)
    return roots


def root_id_for(fen, repetition):
    """Return the identity of a root from its full FEN and repetition count."""
    return f"{fen}|{repetition}"


def split_for(position_key):
    """Return the split assigned to a position key."""
    bucket = random.Random(position_key).randrange(SPLIT_MODULUS)
    return "val" if bucket == VAL_REMAINDER else "train"


def _position_key(fen):
    return " ".join(fen.split()[:4])


def _claimable(record):
    """Whether a root record's game state allows a draw claim."""
    halfmove = int(record["fen"].split()[4])
    return record["repetition"] >= 3 or halfmove >= CLAIM_HALFMOVE


def _pool_priority(seed, root_id, domain="pool"):
    return random.Random(f"{seed}:{root_id}:{domain}").getrandbits(64)


def select_pool(records, train_roots, val_roots, claimable_share, seed):
    """Select reproducible split quotas, optionally stratified by draw claims."""
    if claimable_share is not None and not 0.0 <= claimable_share <= 1.0:
        raise CorpusError("claimable_share must lie in [0, 1]")
    if train_roots < 0 or val_roots < 0 or train_roots + val_roots == 0:
        raise CorpusError("Pool sizes must be nonnegative and not both zero")
    targets = {}
    for split, size in (("train", train_roots), ("val", val_roots)):
        if claimable_share is None:
            targets[split, None] = size
            continue
        claimable = round(size * claimable_share)
        targets[split, True] = claimable
        targets[split, False] = size - claimable
    heaps = {stratum: [] for stratum in targets}
    available = Counter()
    for sequence, record in enumerate(records):
        stratum = (
            record["split"],
            _claimable(record) if claimable_share is not None else None,
        )
        available[stratum] += 1
        target = targets[stratum]
        if target == 0:
            continue
        entry = (
            -_pool_priority(seed, record["root_id"]),
            sequence,
            record["root_id"],
            record,
        )
        heap = heaps[stratum]
        if len(heap) < target:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    selected = []
    strata = {}
    for stratum, heap in heaps.items():
        target = targets[stratum]
        supply = available[stratum]
        split, claimable = stratum
        name = f"{split}:" + (
            "all" if claimable is None else "claimable" if claimable else "plain"
        )
        if supply < target:
            raise CorpusError(
                f"Stratum {name} holds {supply} roots, fewer than the requested {target}"
            )
        strata[name] = {"available": supply, "selected": len(heap)}
        for _, _, root_id, record in heap:
            selected.append((_pool_priority(seed, root_id, "pool-order"), record))
    selected.sort(key=lambda item: item[0])
    return [record for _, record in selected], strata


def write_pool(output, records, manifest):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "roots.jsonl"
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    write_manifest(output / "pool.manifest.json", manifest)
