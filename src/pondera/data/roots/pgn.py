"""Replay PGN archives and preserve eligible game positions."""

import hashlib
import io
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

from pondera.chess import _rules as rules
from pondera.data.roots.pool import FIVEFOLD, CorpusError


def verify_archive(path, source):
    with Path(path).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != source.sha256:
        raise CorpusError(f"{path} has sha256 {digest}, expected {source.sha256}")
    return Path(path)


def ensure_archive(directory, source):
    """Download the configured archive once and verify its digest."""
    directory = Path(directory) / source.sha256
    path = directory / Path(urllib.parse.urlsplit(source.url).path).name
    if path.exists():
        return path
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


_PGN_DELIMITERS = re.compile(r"[{}();]")


def _games(stream):
    """Frame PGN games while keeping archive IO outside the chess backend."""
    lines = []
    tags = post_tags = in_comment = False
    variations = 0
    for raw_line in stream:
        line = raw_line.removeprefix("\ufeff")
        if line.startswith("%"):
            continue
        stripped = line.strip()
        if not in_comment and not variations:
            if stripped.startswith("["):
                if post_tags:
                    yield "".join(lines)
                    lines = []
                lines.append(stripped + "\n")
                tags, post_tags = True, False
                continue
            if tags:
                post_tags = True
            if not stripped:
                continue
        lines.append(line)
        cursor = 0
        for delimiter in _PGN_DELIMITERS.finditer(line):
            if not post_tags and not in_comment and not variations:
                post_tags = bool(line[cursor : delimiter.start()].strip())
            cursor = delimiter.end()
            char = delimiter[0]
            if in_comment:
                if char == "}":
                    in_comment = False
            elif char == ";":
                break
            elif char == "{":
                in_comment = True
            elif char == "(":
                variations += 1
            elif char == ")" and variations:
                variations -= 1
        else:
            if not post_tags and not in_comment and not variations:
                post_tags = bool(line[cursor:].strip())
    if lines:
        yield "".join(lines)


def _replayed_games(games, workers):
    if workers == 1:
        yield from map(rules.replay_mainline, games)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while batch := list(islice(games, 128)):
            yield from executor.map(rules.replay_mainline, batch)


def collect(archive, source, workers, index, *, local=False):
    """Stream eligible per-game winners into the deduplication index."""
    started = time.monotonic()
    counts = Counter(
        games_read=0, games_replayed=0, plies=0, terminal=0, fivefold=0, eligible=0
    )
    skipped = Counter()
    with _PgnStream(archive) as reader:
        games = islice(_games(reader._text), source.limit_games)
        for game, (reason, rows, terminal) in enumerate(
            _replayed_games(games, workers)
        ):
            counts["games_read"] += 1
            if reason or not rows:
                skipped[reason or "empty"] += 1
                continue
            counts["games_replayed"] += 1
            counts["plies"] += len(rows)
            # A later visit supplies the counters and ply for the position.
            last = {
                key: (key, repetition, halfmove, fullmove, game, ply)
                for ply, (key, repetition, halfmove, fullmove) in enumerate(rows, 1)
            }
            if terminal:
                del last[rows[-1][0]]
                counts["terminal"] += 1
            eligible = [
                candidate for candidate in last.values() if candidate[1] < FIVEFOLD
            ]
            counts["fivefold"] += len(last) - len(eligible)
            counts["eligible"] += len(eligible)
            index.add_elite(eligible)
        member = reader.member
    retained = index.count_elite()
    return {
        "kind": source.kind,
        "provenance": {
            "archive_url": None if local else source.url,
            "file": member,
        },
        "limit_games": source.limit_games,
        **counts,
        "games_skipped": dict(skipped),
        "duplicates_within_source": counts["eligible"] - retained,
        "retained": retained,
        "elapsed_seconds": time.monotonic() - started,
    }
