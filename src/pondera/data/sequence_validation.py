"""Integrity checks and independent replay validation for Sequence releases."""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import numpy as np

from pondera.chess.codec import BOARD_TOKENS_MAX, encode_board, value_bin
from pondera.chess.symmetry import reflect_uci
from pondera.chess.vocabulary import (
    CHESS_TOKENIZER_VERSION,
    DEPTH,
    IDX_TO_UCI_MOVE,
    MATE_LOSS,
    MATE_WIN,
    MOVE,
    PAD,
    VALUE,
)
from pondera.data.storage import content_sha256, load_manifest, sha256, write_manifest
from pondera.data.trace import (
    CLAIM_DRAW_UCI,
    DRAW_CLAIM_BOUND,
    ROLE_COUNT,
    TRACE_GRAMMAR_VERSION,
    replay,
)
from pondera.data.trace_synthesis import (
    PRESENTATIONS,
    SIDECAR_SCHEMA,
    SIDECAR_SCHEMA_WIDE,
    mate_distance_moves,
)

SEQUENCE_FORMAT_VERSION = 5
TRACE_ARRAYS = ("board_tokens", "trace_tokens", "trace_len", "trace_roles")
STATE_ARRAYS = (
    "state_no_tool",
    "state_tool",
    "state_best",
    "state_basis",
    "state_value",
)
SEQUENCE_ARRAYS = (*TRACE_ARRAYS, *STATE_ARRAYS, "state_count")
# The label region after FINAL is not a thinking boundary; truncation states
# stop at the last thought operation.
FINAL_SUFFIX_OPS = frozenset({"final", "final_evaluate", "final_pv", "end"})
RELEASE_ONLY_METADATA = frozenset(
    {"root_selection", "root_selections", "roots", "root_sets", "sidecars"}
)
ROOT_RECORD_FIELDS = ("split", "fen", "repetition", "turn", "position_key", "source")
SEARCH_FIELDS = frozenset(
    {"depth", "teacher_width", "threads", "hash_mb", "fresh_engine_per_root"}
)
BOARD_FIELDS = frozenset({"board_descent_hazard", "board_frontier_p"})
PRESENTATION_FIELDS = frozenset(
    {
        "zero",
        "full",
        "random",
        "last_root",
        "random_min_fraction",
        "random_max_fraction",
    }
)
SYNTHESIS_FIELDS = frozenset({"confirmatory_repetition", "claim_draw"})
TEACHER_FIELDS = frozenset(
    {"schema_version", "contract", "contract_sha256", "value_mate", "max_ply"}
)
ENGINE_FIELDS = frozenset(
    {"name", "sha256", "instrumentation", "instrumentation_sha256", "control"}
)
ROOT_SELECTION_FIELDS = frozenset(
    {"source", "source_sha256", "algorithm", "seed", "requested", "counts"}
)


def shard_configuration(metadata):
    return {
        key: value
        for key, value in metadata.items()
        if key not in RELEASE_ONLY_METADATA
    }


def _is_hex64(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_int(value, minimum=None):
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    return minimum is None or value >= minimum


def _is_probability(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and 0.0 <= float(value) <= 1.0
    )


def _valid_root_set(identity):
    if not isinstance(identity, dict) or set(identity) != {"rows", "sha256"}:
        return False
    return _is_int(identity["rows"], 1) and _is_hex64(identity["sha256"])


def _release_identity_contract(release, root_count):
    metadata = release.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Release has no synthesis identity")
    required = {
        "teacher",
        "engine",
        "config",
        "search",
        "board",
        "presentation",
        "synthesis",
        "root_selection",
    }
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"Release identity is missing {sorted(missing)[0]}")

    teacher = metadata["teacher"]
    if not isinstance(teacher, dict) or set(teacher) != TEACHER_FIELDS:
        raise ValueError("Release teacher identity is incomplete")
    if teacher["schema_version"] not in (SIDECAR_SCHEMA, SIDECAR_SCHEMA_WIDE):
        raise ValueError("Release teacher schema is unsupported")
    if not isinstance(teacher["contract"], str):
        raise ValueError("Release teacher contract path is invalid")
    if not _is_hex64(teacher["contract_sha256"]):
        raise ValueError("Release teacher contract hash is invalid")
    if not _is_int(teacher["value_mate"], 1) or not _is_int(teacher["max_ply"], 1):
        raise ValueError("Release teacher mate constants are invalid")

    engine = metadata["engine"]
    if not isinstance(engine, dict) or set(engine) != ENGINE_FIELDS:
        raise ValueError("Release engine identity is incomplete")
    if not isinstance(engine["name"], str) or not engine["name"]:
        raise ValueError("Release engine name is invalid")
    if not _is_hex64(engine["sha256"]):
        raise ValueError("Release engine hash is invalid")
    if not isinstance(engine["instrumentation"], str):
        raise ValueError("Release instrumentation path is invalid")
    if not _is_hex64(engine["instrumentation_sha256"]):
        raise ValueError("Release instrumentation hash is invalid")
    control = engine["control"]
    if not isinstance(control, dict) or not _is_hex64(control.get("sha256")):
        raise ValueError("Release control binary identity is invalid")

    config = metadata["config"]
    if (
        not isinstance(config, dict)
        or set(config) != {"path", "sha256"}
        or not isinstance(config["path"], str)
        or not _is_hex64(config["sha256"])
    ):
        raise ValueError("Release configuration identity is invalid")

    search = metadata["search"]
    if not isinstance(search, dict) or set(search) != SEARCH_FIELDS:
        raise ValueError("Release identity has incomplete search settings")
    if not _is_int(search["depth"], 1) or not _is_int(search["hash_mb"], 1):
        raise ValueError("Release search limits are invalid")
    if not _is_int(search["teacher_width"], 1):
        raise ValueError("Release teacher width is invalid")
    if (teacher["schema_version"] == SIDECAR_SCHEMA) != (search["teacher_width"] == 1):
        raise ValueError("Release teacher schema differs from the teacher width")
    if search["threads"] != 1:
        raise ValueError("Release must use one search thread")
    if search["fresh_engine_per_root"] is not True:
        raise ValueError("Release must restart the engine for every root")

    board = metadata["board"]
    if not isinstance(board, dict) or set(board) != BOARD_FIELDS:
        raise ValueError("Release BOARD identity is incomplete")
    if not all(_is_probability(board[name]) for name in BOARD_FIELDS):
        raise ValueError("Release BOARD probabilities are invalid")

    presentation = metadata["presentation"]
    if not isinstance(presentation, dict) or set(presentation) != PRESENTATION_FIELDS:
        raise ValueError("Release presentation identity is incomplete")
    fractions = [presentation[name] for name in ("zero", "full", "random", "last_root")]
    if not all(_is_probability(value) for value in fractions):
        raise ValueError("Release presentation fractions are invalid")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Release presentation fractions do not sum to one")
    if not (
        _is_probability(presentation["random_min_fraction"])
        and _is_probability(presentation["random_max_fraction"])
        and presentation["random_min_fraction"] < presentation["random_max_fraction"]
    ):
        raise ValueError("Release random cut interval is invalid")

    synthesis = metadata["synthesis"]
    if not isinstance(synthesis, dict) or set(synthesis) != SYNTHESIS_FIELDS:
        raise ValueError("Release synthesis identity is incomplete")
    if synthesis["confirmatory_repetition"] != "faithful":
        raise ValueError("Release confirmatory repetitions are not faithful")
    if not isinstance(synthesis["claim_draw"], bool):
        raise ValueError("Release draw claim setting is invalid")

    root_selection = metadata["root_selection"]
    if (
        not isinstance(root_selection, dict)
        or set(root_selection)
        not in (ROOT_SELECTION_FIELDS, ROOT_SELECTION_FIELDS | {"source_counts"})
        or not isinstance(root_selection.get("source"), str)
        or not _is_hex64(root_selection.get("source_sha256"))
        or not isinstance(root_selection.get("algorithm"), str)
        or not _is_int(root_selection.get("seed"))
        or not _is_int(root_selection.get("requested"), 1)
        or not isinstance(root_selection.get("counts"), dict)
        or set(root_selection["counts"]) != {"train", "val"}
        or not all(_is_int(value, 0) for value in root_selection["counts"].values())
    ):
        raise ValueError("Release root selection identity is incomplete")
    source_counts = root_selection.get("source_counts")
    if source_counts is not None and (
        not isinstance(source_counts, dict)
        or set(source_counts) != {"train", "val"}
        or not all(_is_int(value, 0) for value in source_counts.values())
    ):
        raise ValueError("Release source root counts are invalid")

    if ("roots" in metadata) == ("root_sets" in metadata):
        raise ValueError("Release identity needs one roots or root_sets record")
    if "roots" in metadata:
        root_sets = [metadata["roots"]]
    else:
        root_sets = metadata["root_sets"]
        if not isinstance(root_sets, list) or not root_sets:
            raise ValueError("Release root_sets must be a nonempty list")
    if not all(_valid_root_set(root_set) for root_set in root_sets):
        raise ValueError("Release root identity is malformed")
    if sum(root_set["rows"] for root_set in root_sets) != root_count:
        raise ValueError("Release root identity does not match its root count")
    if "sidecars" in metadata:
        sidecars = metadata["sidecars"]
        if not _valid_root_set(sidecars) or sidecars["rows"] != sum(
            root_set["rows"] for root_set in root_sets
        ):
            raise ValueError("Release sidecar identity is malformed")
    for key, value in metadata.items():
        if release.get(key) != value:
            raise ValueError(f"Release metadata {key} differs from its identity")
    return metadata


def validate_release_identity(release):
    complete = release.get("complete_root_ids")
    if (
        not isinstance(complete, list)
        or not complete
        or len(set(complete)) != len(complete)
    ):
        raise ValueError("Release root ids must be nonempty and unique")
    return _release_identity_contract(release, len(complete))


def validate_sequence_arrays(arrays, rows=None):
    """Check the shared array contract without interpreting individual traces."""
    if not arrays:
        raise ValueError("A Sequence dataset needs token arrays")
    row_count = len(next(iter(arrays.values())))
    if rows is not None and row_count != rows:
        raise ValueError("Sequence arrays do not match the declared row count")
    for name, array in arrays.items():
        if len(array) != row_count:
            raise ValueError(f"Sequence array {name} has a different row count")

    boards = arrays.get("board_tokens")
    if boards is not None and (
        boards.ndim != 2
        or boards.dtype != np.int16
        or boards.shape[1] > BOARD_TOKENS_MAX
    ):
        raise ValueError("Board token array does not match its declared format")

    trace = arrays.get("trace_tokens")
    if trace is not None and (trace.ndim != 2 or trace.dtype != np.int16):
        raise ValueError("Trace token array does not match its examples")
    lengths = arrays.get("trace_len")
    if lengths is not None:
        if lengths.shape != (row_count,) or lengths.dtype != np.int32:
            raise ValueError("Trace lengths do not match their examples")
        if trace is not None and trace.ndim == 2:
            if np.any(lengths < 1) or np.any(lengths > trace.shape[1]):
                raise ValueError("Trace lengths fall outside their token array")
    roles = arrays.get("trace_roles")
    if roles is not None:
        if trace is not None and roles.shape != trace.shape:
            raise ValueError("Trace roles do not match their examples")
        if roles.dtype != np.int8 or np.any((roles < 0) | (roles >= ROLE_COUNT)):
            raise ValueError("Trace roles do not match their declared format")

    best = arrays.get("state_best")
    if best is not None and (best.ndim != 2 or best.dtype != np.int32):
        raise ValueError("Truncation-state arrays do not match their examples")
    for name in STATE_ARRAYS:
        array = arrays.get(name)
        if array is not None and (
            array.dtype != np.int32 or (best is not None and array.shape != best.shape)
        ):
            raise ValueError(f"Truncation-state array {name} has inconsistent shape")
    counts = arrays.get("state_count")
    if counts is not None:
        if counts.shape != (row_count,) or counts.dtype != np.int32:
            raise ValueError("Truncation-state counts do not match their examples")
        if best is not None and best.ndim == 2:
            if np.any(counts < 0) or np.any(counts > best.shape[1]):
                raise ValueError("Truncation-state counts fall outside their arrays")


def load_sequence_arrays(directory, names=TRACE_ARRAYS):
    arrays = {
        name: np.load(
            Path(directory) / f"{name}.npy", mmap_mode="r", allow_pickle=False
        )
        for name in names
    }
    validate_sequence_arrays(arrays)
    return arrays


def sequence_data_identity(directory):
    """Verify and fingerprint the files that define Sequence training input."""
    directory = Path(directory)
    manifest_path = directory / (
        "release.json" if (directory / "release.json").exists() else "manifest.json"
    )
    manifest = load_manifest(manifest_path)
    files = {}
    if "shards" in manifest:
        for shard in manifest["shards"]:
            shard_dir = directory / shard["path"]
            manifest_path = shard_dir / "manifest.json"
            manifest_sha = sha256(manifest_path)
            if manifest_sha != shard["sha256"]:
                raise ValueError(f"Shard manifest checksum mismatch: {manifest_path}")
            shard_manifest = load_manifest(manifest_path)
            files[manifest_path.relative_to(directory).as_posix()] = manifest_sha
            filenames = [f"{name}.npy" for name in TRACE_ARRAYS] + ["roots.jsonl"]
            for filename in filenames:
                expected = shard_manifest.get("files", {}).get(filename)
                if expected is None:
                    raise ValueError(
                        f"Shard has no checksum for {filename}: {shard_dir}"
                    )
                path = shard_dir / filename
                actual = sha256(path)
                if actual != expected:
                    raise ValueError(f"Training input checksum mismatch: {path}")
                files[path.relative_to(directory).as_posix()] = actual
    else:
        if not manifest.get("files"):
            raise ValueError("Sequence dataset manifest has no input checksums")
        for filename, expected in manifest["files"].items():
            path = directory / filename
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"Training input checksum mismatch: {path}")
            files[filename] = actual
    return content_sha256({"manifest": manifest, "files": files})


def verify_shard_files(release_dir, shard, expected_metadata=None):
    """Verify a published shard and return its manifest, roots, and arrays."""
    directory = Path(release_dir) / shard["path"]
    manifest_path = directory / "manifest.json"
    if sha256(manifest_path) != shard["sha256"]:
        raise ValueError(f"Shard manifest checksum mismatch: {manifest_path}")
    manifest = load_manifest(manifest_path)
    if expected_metadata is not None:
        for key, value in shard_configuration(expected_metadata).items():
            if manifest.get(key) != value:
                raise ValueError(f"Shard metadata {key} differs from its release")
    if manifest["rows"] != shard["rows"]:
        raise ValueError(f"Shard row count differs from its release: {directory}")
    expected_files = {f"{name}.npy" for name in SEQUENCE_ARRAYS} | {
        "roots.jsonl",
        "champions.jsonl",
    }
    if set(manifest.get("files", {})) != expected_files:
        raise ValueError(f"Shard file inventory is incomplete: {directory}")
    for name, expected in manifest["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Shard file checksum mismatch: {directory / name}")
    roots = [
        json.loads(line)
        for line in (directory / "roots.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(roots) != manifest["rows"]:
        raise ValueError(f"Shard roots do not match its manifest: {directory}")
    arrays = load_sequence_arrays(directory, SEQUENCE_ARRAYS)
    validate_sequence_arrays(arrays, rows=len(roots))
    return manifest, roots, arrays


def _read_champions(directory):
    directory = Path(directory)
    return [
        json.loads(line)
        for line in (directory / "champions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def _move_token(uci, mirrored):
    return MOVE + IDX_TO_UCI_MOVE_REVERSE[reflect_uci(uci) if mirrored else uci]


IDX_TO_UCI_MOVE_REVERSE = {move: index for index, move in IDX_TO_UCI_MOVE.items()}


def _check_champion_round(root_id, entry, board, value_mate, max_ply):
    if set(entry) != {
        "round",
        "depth",
        "move",
        "raw",
        "wdl",
        "mate",
        "value_token",
        "pv",
        "refutations",
        "candidates",
    }:
        raise ValueError(f"Champion record is incomplete: {root_id}")
    round_number = entry["round"]
    if not _is_int(round_number, 1) or not _is_int(entry["depth"], 1):
        raise ValueError(f"Champion round is invalid: {root_id}")
    move = entry["move"]
    if not isinstance(move, str) or chess.Move.from_uci(move) not in board.legal_moves:
        raise ValueError(f"Champion move is illegal: {root_id}")
    pv = entry["pv"]
    if not isinstance(pv, list) or not pv or pv[0] != move:
        raise ValueError(f"Champion principal variation is invalid: {root_id}")
    tracker = chess.Board(board.fen(en_passant="fen"))
    for index, step in enumerate(pv):
        if not isinstance(step, str):
            raise ValueError(f"Champion principal variation has a bad move: {root_id}")
        try:
            tracker.push_uci(step)
        except ValueError as error:
            raise ValueError(
                f"Champion principal variation is illegal: {root_id}"
            ) from error
        if index + 1 < len(pv) and tracker.is_game_over():
            # The search rule, not python-chess's material rule, defines the
            # trace's terminals; the writer re-checks the exact rule on replay.
            continue
    value_token = entry["value_token"]
    if not isinstance(value_token, int) or not VALUE <= value_token < PAD:
        raise ValueError(f"Champion value token is invalid: {root_id}")
    wdl = entry["wdl"]
    mate = entry["mate"]
    if wdl is None:
        if mate is None or not isinstance(mate, int) or mate == 0:
            raise ValueError(f"Champion mate claim is incomplete: {root_id}")
        expected = mate_token_for(mate)
        if value_token != expected:
            raise ValueError(f"Champion mate token disagrees: {root_id}")
        if mate_distance_moves(entry["raw"], value_mate, max_ply) != mate:
            raise ValueError(f"Champion mate distance disagrees: {root_id}")
    else:
        if (
            not isinstance(wdl, list)
            or len(wdl) != 3
            or any(not _is_int(value, 0) for value in wdl)
            or sum(wdl) != 1000
            or mate is not None
        ):
            raise ValueError(f"Champion WDL is invalid: {root_id}")
        expected = VALUE + value_bin((wdl[0] - wdl[2]) / 1000)
        if value_token != expected:
            raise ValueError(f"Champion value token disagrees: {root_id}")
    candidates = entry["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Champion round has no exact candidates: {root_id}")
    seen_moves = set()
    champion_raw = None
    best_raw = None
    for candidate in candidates:
        if set(candidate) != {
            "move",
            "raw",
            "wdl",
            "mate",
            "value_token",
            "order",
            "draw",
        }:
            raise ValueError(f"Champion candidate is incomplete: {root_id}")
        if (
            not isinstance(candidate["move"], str)
            or chess.Move.from_uci(candidate["move"]) not in board.legal_moves
            or not _is_int(candidate["order"], 1)
            or candidate["move"] in seen_moves
        ):
            raise ValueError(f"Champion candidate is invalid: {root_id}")
        seen_moves.add(candidate["move"])
        if not isinstance(candidate["draw"], bool):
            raise ValueError(f"Champion candidate draw flag is invalid: {root_id}")
        if candidate["draw"]:
            if candidate["wdl"] is not None or candidate["mate"] is not None:
                raise ValueError(
                    f"Champion draw candidate carries an engine value: {root_id}"
                )
            expected = VALUE + value_bin(0.0)
        elif candidate["wdl"] is None:
            if candidate["mate"] is None:
                raise ValueError(f"Champion candidate has no value: {root_id}")
            expected = mate_token_for(candidate["mate"])
        else:
            if (
                not isinstance(candidate["wdl"], list)
                or len(candidate["wdl"]) != 3
                or any(not _is_int(value, 0) for value in candidate["wdl"])
                or sum(candidate["wdl"]) != 1000
                or candidate["mate"] is not None
            ):
                raise ValueError(f"Champion candidate WDL is invalid: {root_id}")
            expected = VALUE + value_bin(
                (candidate["wdl"][0] - candidate["wdl"][2]) / 1000
            )
        if candidate["value_token"] != expected:
            raise ValueError(f"Champion candidate token disagrees: {root_id}")
        if candidate["move"] == move:
            champion_raw = candidate["raw"]
        if best_raw is None or candidate["raw"] > best_raw:
            best_raw = candidate["raw"]
    # Equal root values are possible; the engine's rank names one of the tied
    # maximums, and its tie-break is not carried by the trace.
    if champion_raw is None:
        raise ValueError(f"Champion move is not a written candidate: {root_id}")
    if champion_raw != entry["raw"]:
        raise ValueError(f"Champion raw disagrees with its candidate: {root_id}")
    if best_raw != champion_raw:
        raise ValueError(
            f"Champion is not a raw maximum among the candidates: {root_id}"
        )


def mate_token_for(mate):
    if mate >= 0:
        return MATE_WIN + mate
    return MATE_LOSS + (-mate)


def _expected_states(program, tokens, champions, mirrored):
    """Derive the champion visible at each thought boundary from the trace."""
    entries = {entry["round"]: entry for entry in champions["rounds"]}
    states = []
    round_number = 0
    round_candidates = {}
    previous = None
    current = None
    for step in program.steps:
        if step.operation in FINAL_SUFFIX_OPS:
            break
        if step.operation == "depth":
            round_number += 1
            round_candidates = {}
            current = previous
        elif step.operation == "evaluate" and len(step.path) == 1:
            move = step.path[-1]
            entry = entries.get(round_number)
            if entry is not None:
                candidate = next(
                    (
                        candidate
                        for candidate in entry["candidates"]
                        if candidate["move"] == move
                    ),
                    None,
                )
                if candidate is not None:
                    round_candidates[move] = candidate
                    best = max(
                        round_candidates.values(),
                        key=lambda candidate: (
                            candidate["raw"],
                            -candidate["order"],
                            candidate["move"],
                            candidate["value_token"],
                        ),
                    )
                    current = (
                        _move_token(best["move"], mirrored),
                        best["value_token"],
                    )
        elif step.operation == "evaluate" and not step.path:
            entry = entries.get(round_number)
            if entry is not None:
                current = (_move_token(entry["move"], mirrored), entry["value_token"])
                previous = current
        if current is None:
            states.append((-1, -1, round_number))
        else:
            states.append((current[0], current[1], round_number))
    return states


def _validate_shard_rows(
    release_dir,
    shard,
    presentations,
    teacher_width,
    depth,
    value_mate,
    max_ply,
    expected_metadata=None,
    expected_roots=None,
):
    directory = Path(release_dir) / shard["path"]
    manifest, rows, arrays = verify_shard_files(release_dir, shard, expected_metadata)
    champions_rows = _read_champions(directory)
    champions = {row["root_id"]: row for row in champions_rows}
    if len(champions) != len(champions_rows):
        raise ValueError(f"Champion records contain duplicate roots: {directory}")
    identities = []
    counts = Counter()
    state_total = 0
    board_responses = 0
    max_trace = 0
    presentations_seen = Counter()
    suffixes = {}
    roots_seen = set()
    checked_champions = set()
    for index, row in enumerate(rows):
        root_id = row.get("root_id")
        presentation = row.get("presentation")
        if not isinstance(root_id, str) or presentation not in presentations:
            raise ValueError(f"Release row identity is invalid: {directory}")
        if row.get("teacher_width") != teacher_width:
            raise ValueError(f"Release teacher width differs: {root_id}")
        if row.get("depth") != depth:
            raise ValueError(f"Release depth differs: {root_id}")
        champion = champions.get(root_id)
        if champion is None:
            raise ValueError(f"Release row has no champion record: {root_id}")
        if expected_roots is not None:
            expected = expected_roots.get(root_id)
            if expected is None:
                raise ValueError(f"Release contains an unselected root: {root_id}")
            for name in ROOT_RECORD_FIELDS:
                if name in expected and row.get(name) != expected.get(name):
                    raise ValueError(
                        f"Release root {name} differs from its selection: {root_id}"
                    )
        board = chess.Board(row["fen"])
        if board.turn != (row["turn"] == "white"):
            raise ValueError(f"Root turn disagrees with FEN: {root_id}")
        repetition = row["repetition"]
        if not _is_int(repetition, 1) or repetition > 4:
            raise ValueError(f"Root repetition is out of range: {root_id}")
        if root_id not in checked_champions:
            if len(champion.get("rounds", [])) != depth:
                raise ValueError(f"Champion record has the wrong depth: {root_id}")
            for entry in champion["rounds"]:
                _check_champion_round(root_id, entry, board, value_mate, max_ply)
            checked_champions.add(root_id)
        encoded = encode_board(row["fen"], repetition)
        board_tokens = arrays["board_tokens"][index]
        if not np.array_equal(board_tokens[: len(encoded)], encoded) or not np.all(
            board_tokens[len(encoded) :] == PAD
        ):
            raise ValueError(f"Board tokens disagree with FEN: {root_id}")

        length = int(arrays["trace_len"][index])
        tokens = arrays["trace_tokens"][index, :length]
        if not np.all(arrays["trace_tokens"][index, length:] == PAD) or not np.all(
            arrays["trace_roles"][index, length:] == 0
        ):
            raise ValueError(f"Trace padding is invalid: {root_id}")
        program = replay(
            tokens,
            row["fen"],
            repetition,
            teacher_width=teacher_width,
        )
        if not np.array_equal(program.roles, arrays["trace_roles"][index, :length]):
            raise ValueError(f"Trace roles disagree with replay: {root_id}")
        max_trace = max(max_trace, length)

        board_spans = [
            (step.start, step.end)
            for step in program.steps
            if step.operation == "board"
        ]
        board_responses += len(board_spans)
        no_tool_tokens = list(tokens)
        for start, end in reversed(board_spans):
            del no_tool_tokens[start:end]
        no_tool = replay(
            no_tool_tokens,
            row["fen"],
            repetition,
            teacher_width=teacher_width,
        )
        expected_boundaries = []
        operation = 0
        previous_no_tool_end = 0
        for step in program.steps:
            if step.operation == "board":
                expected_boundaries.append((step.end, previous_no_tool_end))
                continue
            no_tool_step = no_tool.steps[operation]
            if (
                step.position != no_tool_step.position
                or step.uci != no_tool_step.uci
                or step.evaluation != no_tool_step.evaluation
                or step.mate != no_tool_step.mate
            ):
                raise ValueError(f"Tool and no-tool traces differ: {root_id}")
            previous_no_tool_end = no_tool_step.end
            if step.operation not in FINAL_SUFFIX_OPS:
                expected_boundaries.append((step.end, no_tool_step.end))
            operation += 1
        if operation != len(no_tool.steps):
            raise ValueError(f"Tool and no-tool operations differ: {root_id}")
        expected_boundaries.sort()

        count = int(arrays["state_count"][index])
        if count != len(expected_boundaries):
            raise ValueError(f"Truncation-state count is invalid: {root_id}")
        actual_boundaries = list(
            zip(
                map(int, arrays["state_tool"][index, :count]),
                map(int, arrays["state_no_tool"][index, :count]),
            )
        )
        if actual_boundaries != expected_boundaries:
            raise ValueError(f"Truncation boundaries are invalid: {root_id}")
        for name in STATE_ARRAYS:
            if not np.all(arrays[name][index, count:] == -1):
                raise ValueError(f"Truncation-state padding is invalid: {root_id}")
        expected_states = _expected_states(
            program, tokens, champion, board.turn == chess.BLACK
        )
        if len(expected_states) != count:
            raise ValueError(f"Truncation-state count is invalid: {root_id}")
        for position, expected_state in enumerate(expected_states):
            actual = (
                int(arrays["state_best"][index, position]),
                int(arrays["state_value"][index, position]),
                int(arrays["state_basis"][index, position]),
            )
            if actual != expected_state:
                raise ValueError(f"Truncation state is invalid: {root_id}")

        final_step = next(step for step in program.steps if step.operation == "final")
        final_index = final_step.start
        suffix = tuple(int(token) for token in tokens[final_index:])
        if root_id in suffixes and suffixes[root_id] != suffix:
            raise ValueError(f"Presentations disagree on the FINAL suffix: {root_id}")
        suffixes[root_id] = suffix
        if row.get("thinking_tokens") != final_index:
            raise ValueError(f"Thinking-token count is invalid: {root_id}")
        rounds = sum(1 for token in tokens[:final_index] if token == DEPTH)
        if row.get("rounds") != rounds:
            raise ValueError(f"Row round count is invalid: {root_id}")
        final_move = program.final_move
        if row.get("final_move") != final_move:
            raise ValueError(f"FINAL move differs from the trace: {root_id}")
        if row.get("claim_draw") != (final_move == CLAIM_DRAW_UCI):
            raise ValueError(f"Draw-claim flag differs from the trace: {root_id}")
        value_token = int(tokens[final_index + 2])
        if row.get("final_value_bin") != value_token - VALUE:
            raise ValueError(f"FINAL value bin differs from the trace: {root_id}")
        mate_distance = None
        mate_index = final_index + 3
        if mate_index < len(tokens) and MATE_WIN <= int(tokens[mate_index]) < PAD:
            token = int(tokens[mate_index])
            mate_distance = (
                -(token - MATE_LOSS) if token >= MATE_LOSS else token - MATE_WIN
            )
        if row.get("mate_distance") != mate_distance:
            raise ValueError(f"FINAL mate distance differs from the trace: {root_id}")
        deepest = champion["rounds"][-1]
        if deepest["round"] != champion["rounds"].__len__():
            raise ValueError(f"Champion rounds are not contiguous: {root_id}")
        if final_move == CLAIM_DRAW_UCI:
            claimable = repetition >= 3 or board.halfmove_clock >= 100
            if not claimable or deepest["raw"] > DRAW_CLAIM_BOUND:
                raise ValueError(f"Draw claim is not justified: {root_id}")
        elif final_move != deepest["move"]:
            raise ValueError(f"FINAL move differs from the deepest champion: {root_id}")
        if deepest["mate"] != mate_distance:
            raise ValueError(
                f"FINAL mate distance differs from the champion: {root_id}"
            )
        if deepest["wdl"] is None:
            expected_bin = 19 if deepest["mate"] > 0 else 0
        else:
            expected_bin = value_bin((deepest["wdl"][0] - deepest["wdl"][2]) / 1000)
        if row.get("final_value_bin") != expected_bin:
            raise ValueError(f"FINAL value bin differs from the champion: {root_id}")
        pv_start = final_index + (4 if mate_distance is not None else 3)
        pv_tokens = [int(token) for token in tokens[pv_start:-1]]
        mirrored = board.turn == chess.BLACK
        actual_pv = [
            reflect_uci(IDX_TO_UCI_MOVE[token - MOVE])
            if mirrored
            else IDX_TO_UCI_MOVE[token - MOVE]
            for token in pv_tokens
        ]
        expected_pv = [] if final_move == CLAIM_DRAW_UCI else deepest["pv"][1:]
        if actual_pv != expected_pv:
            raise ValueError(f"FINAL variation differs from the champion: {root_id}")

        identities.append((root_id, presentation))
        counts[row["split"]] += 1
        counts[presentation] += 1
        state_total += count
        presentations_seen[root_id] += 1
        roots_seen.add(root_id)
    if roots_seen != set(champions):
        raise ValueError(f"Shard rows do not cover its champion records: {directory}")
    if any(count != len(presentations) for count in presentations_seen.values()):
        raise ValueError(f"A shard root is missing a presentation: {directory}")
    for champion in champions.values():
        if len(champion["rounds"]) != depth:
            raise ValueError(f"Champion record has the wrong depth: {directory}")
        if [entry["round"] for entry in champion["rounds"]] != list(
            range(1, depth + 1)
        ):
            raise ValueError(f"Champion rounds are not contiguous: {directory}")
    return {
        "identities": identities,
        "counts": counts,
        "states": state_total,
        "board_responses": board_responses,
        "max_trace": max_trace,
    }


def validate_release(directory, roots=None, workers=8):
    """Validate every file and replay every row of a complete sharded release."""
    if workers < 1:
        raise ValueError("Validation workers must be positive")
    directory = Path(directory)
    release = load_manifest(directory / "release.json")
    if not release:
        raise FileNotFoundError(directory / "release.json")
    expected_metadata = {
        "format_version": SEQUENCE_FORMAT_VERSION,
        "grammar_version": TRACE_GRAMMAR_VERSION,
        "tokenizer_version": CHESS_TOKENIZER_VERSION,
        "kind": "teacher-sequence",
        "role_count": ROLE_COUNT,
    }
    for key, value in expected_metadata.items():
        if release.get(key) != value:
            raise ValueError(f"Release metadata {key} is unsupported")
    complete = release["complete_root_ids"]
    metadata = validate_release_identity(release)
    teacher = metadata["teacher"]
    search = metadata["search"]
    expected_shard_metadata = {**expected_metadata, **shard_configuration(metadata)}
    expected = None
    if roots is not None:
        root_records = list(roots)
        expected = {row["root_id"]: row for row in root_records}
        if len(expected) != len(root_records):
            raise ValueError("External root records contain duplicate root ids")
        if "roots" in metadata:
            identity = metadata["roots"]
            if (
                len(root_records) != identity["rows"]
                or content_sha256(root_records) != identity["sha256"]
            ):
                raise ValueError("External root records do not match the release hash")
    seen = set()
    seen_roots = set()
    counts = Counter()
    states = 0
    board_responses = 0
    max_trace = 0

    def validate(shard):
        return _validate_shard_rows(
            directory,
            shard,
            PRESENTATIONS,
            search["teacher_width"],
            search["depth"],
            teacher["value_mate"],
            teacher["max_ply"],
            expected_shard_metadata,
            expected,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(validate, release["shards"]):
            for root_id, presentation in result["identities"]:
                key = (root_id, presentation)
                if key in seen:
                    raise ValueError(f"Duplicate release row: {key}")
                seen.add(key)
                seen_roots.add(root_id)
            counts.update(result["counts"])
            states += result["states"]
            board_responses += result["board_responses"]
            max_trace = max(max_trace, result["max_trace"])
    if seen_roots != set(complete) or len(seen) != len(complete) * len(PRESENTATIONS):
        raise ValueError("Release rows do not match its roots and presentations")
    if expected is not None:
        wanted = {
            (root_id, presentation)
            for root_id in expected
            for presentation in PRESENTATIONS
        }
        if seen != wanted or set(complete) != set(expected):
            raise ValueError("Release does not contain exactly the selected roots")
    result = {
        "ok": True,
        "roots": len(complete),
        "rows": len(seen),
        "states": states,
        "board_responses": board_responses,
        "max_trace": max_trace,
        "counts": dict(counts),
        "shards": len(release["shards"]),
    }
    write_manifest(directory / "validation.json", result)
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--roots", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    roots = None
    if args.roots is not None:
        with args.roots.open(encoding="utf-8") as handle:
            roots = [json.loads(line) for line in handle if line.strip()][
                args.start : args.end
            ]
    result = validate_release(args.release, roots, args.workers)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
