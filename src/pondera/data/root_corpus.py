"""Build a root pool from Lichess Elite games with real in-game history.

The pool is streamed from a pinned monthly Lichess Elite archive. Every game is
replayed with python-chess and the position after each move is recorded together
with the number of times that position has occurred in the game so far,
including the starting position. The starting position is never emitted as a
root, but it counts toward repetition so a position that returns to it reports
the true in-game occurrence count.

Within one game a position key keeps only its last occurrence. Every later
occurrence has a higher repetition count, and its halfmove clock cannot be
smaller, so the last occurrence carries the maxima. Across games the same
position key keeps the record with the higher repetition count; on a tie the
first record in file order wins. A position seen five or more times in one game
is a fivefold-repetition drawn position and is not emitted; such records are
counted instead. Terminal positions (checkmate or stalemate) are excluded
because a root must have at least one legal move.

``position_key`` is the first four FEN fields. ``root_id`` is the lowercase hex
SHA-256 of ``"<fen>|<repetition>"``. Splits are assigned from the position key
alone: the first eight bytes of its SHA-256, read big-endian, modulo 65; bucket
0 is validation and the other 64 buckets are training. That reproduces the
8192/532480 validation share of the previous root release, and the same key
always lands in the same split.

Position keys use python-chess's default FEN en-passant field, which names the
en-passant square only when a legal en-passant capture exists.

Archives may be ``.zip`` (the Lichess Elite distribution), ``.zst`` (decoded
with the system ``zstd`` binary), or plain ``.pgn``. No decompression module is
imported into the package.
"""

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from statistics import median
from typing import NamedTuple

import chess
import chess.pgn

from pondera.data.storage import sha256, write_manifest

SOURCE_MONTH = "2025-11"
SOURCE_FILE = f"lichess_elite_{SOURCE_MONTH}.zip"
SOURCE_URL = f"https://database.nikonoel.fr/{SOURCE_FILE}"
SOURCE_SHA256 = "1ffa577aa0a1352fa04c9a890a6706049a3a79edc52810b136758d1e2a1a2eab"
SPLIT_MODULUS = 65
VAL_REMAINDER = 0
FIVEFOLD = 5
CLAIM_HALFMOVE = 100
FULLMOVE_PHASES = (
    ("1-10", 1, 10),
    ("11-20", 11, 20),
    ("21-30", 21, 30),
    ("31-40", 31, 40),
    ("41-60", 41, 60),
    ("61-100", 61, 100),
    ("101+", 101, None),
)


class CorpusError(RuntimeError):
    """A corpus build that cannot be reproduced from its inputs."""


class _Candidate(NamedTuple):
    repetition: int
    halfmove: int
    fullmove: int
    game: int
    ply: int


def root_id_for(fen, repetition):
    """Return the identity of a root from its full FEN and repetition count."""
    return hashlib.sha256(f"{fen}|{repetition}".encode()).hexdigest()


def split_for(position_key):
    """Return the split assigned to a position key."""
    digest = hashlib.sha256(position_key.encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % SPLIT_MODULUS
    return "val" if bucket == VAL_REMAINDER else "train"


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


def _position_key(fen):
    return " ".join(fen.split()[:4])


def ensure_archive(directory):
    """Download the pinned archive once and verify its digest."""
    directory = Path(directory)
    path = directory / SOURCE_FILE
    if path.exists():
        digest = sha256(path)
        if digest != SOURCE_SHA256:
            raise CorpusError(f"{path} has sha256 {digest}, expected {SOURCE_SHA256}")
        return path
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    digest = hashlib.sha256()
    request = urllib.request.Request(
        SOURCE_URL, headers={"User-Agent": "pondera-root-corpus/1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as handle:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != SOURCE_SHA256:
            raise CorpusError(
                f"Downloaded {SOURCE_URL} with sha256 {digest.hexdigest()}, "
                f"expected {SOURCE_SHA256}"
            )
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


def build_corpus(archive, output, limit_games=None, report=None, report_every=20000):
    """Replay an archive and write ``roots.jsonl`` plus its manifest."""
    if limit_games is not None and limit_games < 1:
        raise CorpusError("limit_games must be positive")
    archive = Path(archive)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    games_read = 0
    games_replayed = 0
    skipped = Counter()
    occurrences = 0
    unique_per_game = 0
    skipped_repetition = 0
    skipped_terminal = 0
    duplicates = 0
    replacements = 0
    raw_repetitions = Counter()
    game_plies = []
    pool = {}
    with _PgnStream(archive) as source:
        stream = source._text
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
            if report is not None and report_every and games_read % report_every == 0:
                report(
                    {
                        "games_read": games_read,
                        "games_replayed": games_replayed,
                        "pool_keys": len(pool),
                        "elapsed_seconds": round(time.perf_counter() - started, 1),
                    }
                )
        member = source.member
    roots_path = output / "roots.jsonl"
    temporary = roots_path.with_name(f".{roots_path.name}.tmp")
    splits = Counter()
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for key, candidate in pool.items():
                fen = f"{key} {candidate.halfmove} {candidate.fullmove}"
                split = split_for(key)
                splits[split] += 1
                record = {
                    "root_id": root_id_for(fen, candidate.repetition),
                    "split": split,
                    "fen": fen,
                    "repetition": candidate.repetition,
                    "turn": "white" if key.split()[1] == "w" else "black",
                    "position_key": key,
                    "source": {
                        "file": member,
                        "game": candidate.game,
                        "ply": candidate.ply,
                    },
                }
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        temporary.replace(roots_path)
    finally:
        temporary.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started
    pinned = archive.name == SOURCE_FILE
    source = {
        "file": archive.name,
        "sha256": sha256(archive),
        "bytes": archive.stat().st_size,
        "member": member,
    }
    if pinned:
        source.update(
            {
                "database": "Lichess Elite Database",
                "month": SOURCE_MONTH,
                "url": SOURCE_URL,
                "pinned_sha256": SOURCE_SHA256,
            }
        )
    manifest = {
        "kind": "elite-root-corpus",
        "format_version": 1,
        "source": source,
        "build": {
            "limit_games": limit_games,
            "games_read": games_read,
            "games_replayed": games_replayed,
            "games_skipped": dict(sorted(skipped.items())),
            "elapsed_seconds": round(elapsed, 3),
        },
        "positions": {
            "occurrences": occurrences,
            "unique_per_game": unique_per_game,
            "pool_keys": len(pool),
            "duplicates_merged": duplicates,
            "replaced_by_higher_repetition": replacements,
            "skipped_repetition_ge_5": skipped_repetition,
            "skipped_terminal": skipped_terminal,
        },
        "splits": {
            "train": splits.get("train", 0),
            "val": splits.get("val", 0),
        },
        "profile": _profile(pool, raw_repetitions, game_plies),
        "root_id": 'sha256("<fen>|<repetition>")',
        "split_rule": (
            "sha256(position_key)[:8] big-endian modulo 65; bucket 0 is val"
        ),
        "within_game_dedup": "last occurrence, which carries the maximum count",
        "cross_game_dedup": (
            "higher repetition wins; a tie keeps the first record in file order"
        ),
        "files": {
            "roots.jsonl": {
                "rows": len(pool),
                "bytes": roots_path.stat().st_size,
                "sha256": sha256(roots_path),
            }
        },
    }
    write_manifest(output / "roots.manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path(".local/data/roots-elite"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--limit-games", type=int)
    args = parser.parse_args()
    archive = args.archive or ensure_archive(args.output)
    manifest = build_corpus(
        archive,
        args.output,
        args.limit_games,
        report=lambda progress: print(json.dumps(progress), flush=True),
    )
    print(
        json.dumps(
            {
                "roots": manifest["files"]["roots.jsonl"]["rows"],
                "splits": manifest["splits"],
                "elapsed_seconds": manifest["build"]["elapsed_seconds"],
                "roots_sha256": manifest["files"]["roots.jsonl"]["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
