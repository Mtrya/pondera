"""Read PGN archives, preserve game history, and select stratified roots."""

import hashlib
import io
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from statistics import median
from typing import NamedTuple

import chess
import chess.pgn

from pondera.data.roots.pool import (
    CLAIM_HALFMOVE,
    FIVEFOLD,
    CorpusError,
    _position_key,
    root_id_for,
    select_pool,
    split_for,
)

FULLMOVE_PHASES = (
    ("1-10", 1, 10),
    ("11-20", 11, 20),
    ("21-30", 21, 30),
    ("31-40", 31, 40),
    ("41-60", 41, 60),
    ("61-100", 61, 100),
    ("101+", 101, None),
)


class _Candidate(NamedTuple):
    repetition: int
    halfmove: int
    fullmove: int
    game: int
    ply: int


def _halfmove_bucket(value):
    if value >= CLAIM_HALFMOVE:
        return f"{CLAIM_HALFMOVE}+"
    lower = value // 10 * 10
    return f"{lower}-{lower + 9}"


def _fullmove_phase(value):
    for name, lower, upper in FULLMOVE_PHASES:
        if value >= lower and (upper is None or value <= upper):
            return name
    return FULLMOVE_PHASES[-1][0]


def _share(count, total):
    return round(count / total, 6) if total else 0.0


def verify_archive(path, source):
    with Path(path).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != source.sha256:
        raise CorpusError(f"{path} has sha256 {digest}, expected {source.sha256}")
    return Path(path)


def ensure_archive(directory, source):
    """Download the configured archive once and verify its digest."""
    directory = Path(directory)
    path = directory / Path(urllib.parse.urlsplit(source.url).path).name
    if path.exists():
        return verify_archive(path, source)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    request = urllib.request.Request(
        source.url, headers={"User-Agent": "pondera-root-pool/1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        verify_archive(temporary, source)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


class _PgnStream:
    def __init__(self, archive):
        self.archive = Path(archive)
        self.member = self.archive.name
        self._handle = None
        self._process = None
        self._zip = None
        self._text = None

    def __enter__(self):
        suffix = self.archive.suffix.lower()
        if suffix == ".zip":
            self._zip = zipfile.ZipFile(self.archive)
            members = [
                name for name in self._zip.namelist() if name.lower().endswith(".pgn")
            ]
            if len(members) != 1:
                self._zip.close()
                raise CorpusError(
                    f"{self.archive} must contain exactly one PGN member, "
                    f"found {members}"
                )
            self.member = members[0]
            self._handle = self._zip.open(self.member)
        elif suffix == ".zst":
            executable = shutil.which("zstd")
            if executable is None:
                raise CorpusError("Reading a .zst archive requires the zstd binary")
            self._process = subprocess.Popen(
                [executable, "-dc", "--", str(self.archive)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._handle = self._process.stdout
        elif suffix in (".pgn", ".txt"):
            self._handle = self.archive.open("rb")
        else:
            raise CorpusError(f"Unsupported archive type: {self.archive}")
        self._text = io.TextIOWrapper(
            self._handle, encoding="utf-8", errors="replace", newline=""
        )
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._text.close()
        if self._zip is not None:
            self._zip.close()
        if self._process is not None:
            stderr = self._process.stderr.read().decode("utf-8", "replace")
            self._process.stderr.close()
            code = self._process.wait()
            # Closing the pipe early after a game limit sends SIGPIPE.
            if code != 0 and code != -13 and exc_type is None:
                raise CorpusError(f"zstd failed with status {code}: {stderr.strip()}")
        return False


def _replay_game(game, game_index):
    """Return per-key candidates for one game, plus its ply count."""
    board = game.board()
    counts = {_position_key(board.fen()): 1}
    last = {}
    plies = 0
    try:
        for ply, move in enumerate(game.mainline_moves(), 1):
            board.push(move)
            fen = board.fen()
            key = _position_key(fen)
            count = counts.get(key, 0) + 1
            counts[key] = count
            last[key] = _Candidate(
                count,
                board.halfmove_clock,
                board.fullmove_number,
                game_index,
                ply,
            )
            plies = ply
    except ValueError:
        return None
    if plies and not board.legal_moves:
        del last[_position_key(board.fen())]
        return last, plies, True
    return last, plies, False


def _game_is_replayable(game):
    if game.errors:
        return "illegal"
    if game.headers.get("Variant", "Standard").lower() != "standard":
        return "variant"
    if "FEN" in game.headers or "SetUp" in game.headers:
        return "setup"
    return None


def _profile(pool, raw_repetitions, game_plies):
    total = len(pool)
    repetitions = Counter()
    raw = Counter()
    halfmoves = Counter()
    phases = Counter()
    claimable = repetition_ge_3 = halfmove_ge_100 = 0
    for candidate in pool.values():
        repetitions[str(candidate.repetition)] += 1
        halfmoves[_halfmove_bucket(candidate.halfmove)] += 1
        phases[_fullmove_phase(candidate.fullmove)] += 1
        if candidate.repetition >= 3:
            repetition_ge_3 += 1
        if candidate.halfmove >= CLAIM_HALFMOVE:
            halfmove_ge_100 += 1
        if candidate.repetition >= 3 or candidate.halfmove >= CLAIM_HALFMOVE:
            claimable += 1
    for repetition, count in raw_repetitions.items():
        raw["5+" if repetition >= FIVEFOLD else str(repetition)] += count
    repetition_profile = {
        str(value): repetitions.get(str(value), 0) for value in range(1, 5)
    }
    halfmove_profile = {}
    for lower in range(0, CLAIM_HALFMOVE, 10):
        label = f"{lower}-{lower + 9}"
        halfmove_profile[label] = halfmoves.get(label, 0)
    halfmove_profile[f"{CLAIM_HALFMOVE}+"] = halfmoves.get(f"{CLAIM_HALFMOVE}+", 0)
    phase_profile = {name: phases.get(name, 0) for name, _, _ in FULLMOVE_PHASES}
    plies = sorted(game_plies)
    positions_per_game = {
        "count": len(plies),
        "min": plies[0] if plies else 0,
        "p10": plies[len(plies) // 10] if plies else 0,
        "median": median(plies) if plies else 0,
        "mean": round(sum(plies) / len(plies), 3) if plies else 0.0,
        "p90": plies[len(plies) * 9 // 10] if plies else 0,
        "p99": plies[len(plies) * 99 // 100] if plies else 0,
        "max": plies[-1] if plies else 0,
        "total": sum(plies),
    }
    return {
        "roots": total,
        "repetition": {
            "counts": repetition_profile,
            "shares": {
                name: _share(count, total) for name, count in repetition_profile.items()
            },
            "repetition_ge_2": _share(
                sum(
                    count
                    for name, count in repetition_profile.items()
                    if int(name) >= 2
                ),
                total,
            ),
        },
        "claimable": {
            "count": claimable,
            "share": _share(claimable, total),
            "repetition_ge_3": repetition_ge_3,
            "halfmove_ge_100": halfmove_ge_100,
        },
        "halfmove": {
            "counts": halfmove_profile,
            "shares": {
                name: _share(count, total) for name, count in halfmove_profile.items()
            },
        },
        "fullmove_phase": {
            "counts": phase_profile,
            "shares": {
                name: _share(count, total) for name, count in phase_profile.items()
            },
        },
        "game_records_by_repetition": dict(sorted(raw.items())),
        "positions_per_game": positions_per_game,
    }


def build_pool(
    archive,
    train_roots,
    val_roots,
    claimable_share,
    seed,
    *,
    limit_games=None,
    source=None,
    exclude=(),
):
    """Stream an archive directly into the requested stratified root pool."""
    if limit_games is not None and limit_games < 1:
        raise CorpusError("limit_games must be positive")
    if not 0.0 <= claimable_share <= 1.0:
        raise CorpusError("claimable_share must lie in [0, 1]")
    if train_roots < 0 or val_roots < 0 or train_roots + val_roots == 0:
        raise CorpusError("Pool sizes must be nonnegative and not both zero")
    archive = Path(archive)
    started = time.perf_counter()
    games_read = games_replayed = occurrences = unique_per_game = 0
    skipped = Counter()
    skipped_repetition = skipped_terminal = duplicates = replacements = 0
    raw_repetitions = Counter()
    game_plies = []
    pool = {}
    with _PgnStream(archive) as reader:
        stream = reader._text
        while limit_games is None or games_read < limit_games:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            games_read += 1
            reason = _game_is_replayable(game)
            if reason is not None:
                skipped[reason] += 1
                continue
            replayed = _replay_game(game, games_read - 1)
            if replayed is None:
                skipped["replay"] += 1
                continue
            candidates, plies, terminal = replayed
            if not plies:
                skipped["empty"] += 1
                continue
            if terminal:
                skipped_terminal += 1
            games_replayed += 1
            game_plies.append(plies)
            unique_per_game += len(candidates)
            occurrences += plies
            for candidate in candidates.values():
                raw_repetitions[candidate.repetition] += 1
            for key, candidate in candidates.items():
                if candidate.repetition >= FIVEFOLD:
                    skipped_repetition += 1
                    continue
                existing = pool.get(key)
                if existing is None:
                    pool[key] = candidate
                elif candidate.repetition > existing.repetition:
                    pool[key] = candidate
                    replacements += 1
                else:
                    duplicates += 1
        member = reader.member

    records = []
    for key, candidate in pool.items():
        fen = f"{key} {candidate.halfmove} {candidate.fullmove}"
        records.append(
            {
                "root_id": root_id_for(fen, candidate.repetition),
                "split": split_for(key),
                "fen": fen,
                "repetition": candidate.repetition,
                "turn": "white" if key.split()[1] == "w" else "black",
                "position_key": key,
                "source": {
                    **(source or {}),
                    "file": member,
                    "game": candidate.game,
                    "ply": candidate.ply,
                },
            }
        )
    excluded = set(exclude)
    eligible = (
        [record for record in records if record["position_key"] not in excluded]
        if excluded
        else records
    )
    selected, strata = select_pool(
        eligible, train_roots, val_roots, claimable_share, seed
    )
    manifest = {
        "kind": "elite-root-pool",
        "format_version": 2,
        "seed": seed,
        "parameters": {
            "train": train_roots,
            "val": val_roots,
            "claimable_share": claimable_share,
        },
        "source": {
            **(source or {}),
            "file": archive.name,
            "path": str(archive),
            "member": member,
        },
        "build": {
            "limit_games": limit_games,
            "games_read": games_read,
            "games_replayed": games_replayed,
            "games_skipped": dict(sorted(skipped.items())),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
        "positions": {
            "occurrences": occurrences,
            "unique_per_game": unique_per_game,
            "pool_keys": len(pool),
            "duplicates_merged": duplicates,
            "replaced_by_higher_repetition": replacements,
            "skipped_repetition_ge_5": skipped_repetition,
            "skipped_terminal": skipped_terminal,
            "excluded_positions": len(records) - len(eligible),
        },
        "strata": strata,
        "profile": _profile(pool, raw_repetitions, game_plies),
    }
    return selected, manifest
