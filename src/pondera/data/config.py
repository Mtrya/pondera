"""Shared recipe for root preparation, trace curation, and merged encoding."""

import math
import re
import tomllib
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("configs/data.toml")
PRESENTATION_NAMES = ("zero", "full", "random")


def _integer(section, name, value, minimum=None):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"[{section}] {name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"[{section}] {name} must be at least {minimum}")
    return value


def _probability(section, name, value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"[{section}] {name} must be a number")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"[{section}] {name} must lie in [0, 1]")
    return value


def _distribution(section, name, value, key_parser):
    if isinstance(value, (int, str)) and not isinstance(value, bool):
        value = {value: 1.0}
    if not isinstance(value, dict) or not value:
        raise ValueError(f"[{section}] {name} must be a value or a nonempty table")
    result = {}
    for raw_key, probability in value.items():
        key = key_parser(raw_key)
        if key in result:
            raise ValueError(f"[{section}] {name} repeats a key")
        result[key] = _probability(section, f"{name}.{raw_key}", probability)
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-9):
        raise ValueError(f"[{section}] {name} probabilities must sum to 1")
    return tuple(sorted(result.items(), key=lambda item: str(item[0])))


def _joint_distribution(value):
    if not isinstance(value, list) or not value:
        raise ValueError("[search] joint must be a nonempty array of tables")
    result = {}
    for entry in value:
        entry = _table(entry, "search.joint", {"width", "depth", "probability"})
        width = entry["width"]
        if width != "full":
            width = _integer("search.joint", "width", width, 1)
        depth = _integer("search.joint", "depth", entry["depth"], 1)
        outcome = (width, depth)
        if outcome in result:
            raise ValueError("[search] joint repeats a width/depth pair")
        result[outcome] = _probability(
            "search.joint", "probability", entry["probability"]
        )
    if not math.isclose(sum(result.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("[search] joint probabilities must sum to 1")
    return tuple(sorted(result.items(), key=lambda item: (str(item[0][0]), item[0][1])))


def _presentation_key(value):
    if value not in PRESENTATION_NAMES:
        raise ValueError(f"presentation must be one of {PRESENTATION_NAMES}")
    return value


@dataclass(frozen=True)
class PgnSourceConfig:
    name: str
    kind: str
    url: str
    sha256: str
    train: int
    val: int
    claimable_share: float
    limit_games: int | None


@dataclass(frozen=True)
class PuzzleSourceConfig:
    name: str
    kind: str
    repo_id: str
    revision: str
    train: int
    val: int
    batch_size: int


@dataclass(frozen=True)
class SearchConfig:
    hash_mb: int
    timeout_seconds: int
    syzygy_path: str | None
    syzygy_probe_limit: int


@dataclass(frozen=True)
class SynthesisConfig:
    claim_draw: bool
    board_descent_hazard: float
    board_frontier_p: float
    random_min_fraction: float
    random_max_fraction: float


@dataclass(frozen=True)
class ReleaseConfig:
    shard_size: int
    workers: int
    start_shard: int
    end_shard: int | None


@dataclass(frozen=True)
class SamplingConfig:
    joint: tuple
    presentations: tuple

    def identity(self):
        return {
            "joint": [
                {"width": width, "depth": depth, "probability": probability}
                for (width, depth), probability in self.joint
            ],
            "presentations": dict(self.presentations),
        }


@dataclass(frozen=True)
class MergeConfig:
    vocab_size: int
    fit_split: str
    holdout_fraction: float
    max_roots: int | None
    min_count: int
    per_length_limit: int
    selection_stride: int
    block: int | None
    max_rounds: int
    top: int
    batch_size: int
    workers: int


@dataclass(frozen=True)
class DataConfig:
    seed: int
    sources: tuple
    search: SearchConfig
    synthesis: SynthesisConfig
    release: ReleaseConfig
    sampling: SamplingConfig
    merge: MergeConfig

    def search_dict(self):
        return {
            "hash_mb": self.search.hash_mb,
            "timeout_seconds": self.search.timeout_seconds,
            "syzygy_probe_limit": (
                self.search.syzygy_probe_limit if self.search.syzygy_path else 0
            ),
        }

    def board_dict(self):
        return {
            "board_descent_hazard": self.synthesis.board_descent_hazard,
            "board_frontier_p": self.synthesis.board_frontier_p,
        }

    def presentation_dict(self):
        probabilities = dict(self.sampling.presentations)
        return {
            "zero": probabilities.get("zero", 0.0),
            "full": probabilities.get("full", 0.0),
            "random": probabilities.get("random", 0.0),
            "last_root": 0.0,
            "random_min_fraction": self.synthesis.random_min_fraction,
            "random_max_fraction": self.synthesis.random_max_fraction,
        }

    def synthesis_dict(self):
        return {
            "confirmatory_repetition": "faithful",
            "claim_draw": self.synthesis.claim_draw,
        }


def parse_recipe(values):
    values = _table(values, "recipe", {"joint", "presentations"})
    return SamplingConfig(
        joint=_joint_distribution(values["joint"]),
        presentations=_distribution(
            "recipe", "presentations", values["presentations"], _presentation_key
        ),
    )


def _table(values, name, required, optional=()):
    if not isinstance(values, dict):
        raise ValueError(f"[{name}] must be a table")
    unknown = set(values) - set(required) - set(optional)
    if unknown:
        raise ValueError(f"Unknown config key in [{name}]: {sorted(unknown)[0]}")
    missing = set(required) - set(values)
    if missing:
        raise ValueError(f"Missing config key in [{name}]: {sorted(missing)[0]}")
    return values


def _sources(values):
    if not isinstance(values, dict) or not values:
        raise ValueError("[sources] must contain named source tables")
    sources = []
    for name, raw in values.items():
        section = f"sources.{name}"
        if not isinstance(raw, dict):
            raise ValueError(f"[{section}] must be a table")
        kind = raw.get("kind")
        common = {"kind", "train", "val"}
        if kind == "pgn":
            raw = _table(
                raw,
                section,
                common | {"url", "sha256", "claimable_share"},
                {"limit_games"},
            )
        elif kind == "lichess_puzzles":
            raw = _table(raw, section, common | {"repo_id", "revision", "batch_size"})
        else:
            raise ValueError(f"Unknown source kind in [{section}]: {kind}")
        train = _integer(section, "train", raw["train"], 0)
        val = _integer(section, "val", raw["val"], 0)
        if kind == "pgn":
            url, digest = raw["url"], raw["sha256"]
            if (
                not isinstance(url, str)
                or urllib.parse.urlsplit(url).scheme not in {"https", "http", "file"}
                or not Path(urllib.parse.urlsplit(url).path).name
            ):
                raise ValueError(
                    f"[{section}] url must identify a downloadable archive"
                )
            if (
                not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
            ):
                raise ValueError(f"[{section}] sha256 must be a SHA256 digest")
            source = PgnSourceConfig(
                name,
                kind,
                url,
                digest.lower(),
                train,
                val,
                _probability(section, "claimable_share", raw["claimable_share"]),
                _integer(section, "limit_games", raw["limit_games"], 1)
                if "limit_games" in raw
                else None,
            )
        else:
            repo_id, revision = raw["repo_id"], raw["revision"]
            if not isinstance(repo_id, str) or not repo_id:
                raise ValueError(f"[{section}] repo_id must be a nonempty string")
            if (
                not isinstance(revision, str)
                or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None
            ):
                raise ValueError(
                    f"[{section}] revision must be an immutable Hub commit"
                )
            source = PuzzleSourceConfig(
                name,
                kind,
                repo_id,
                revision.lower(),
                train,
                val,
                _integer(section, "batch_size", raw["batch_size"], 1),
            )
        sources.append(source)
    return tuple(sources)


def load_config(path=DEFAULT_CONFIG_PATH):
    path = Path(path)
    with path.open("rb") as handle:
        values = tomllib.load(handle)
    allowed = {"seed", "sources", "search", "synthesis", "release", "merge"}
    if set(values) - allowed:
        raise ValueError(f"Unknown config section: {sorted(set(values) - allowed)[0]}")
    for section in allowed - {"seed"}:
        if section not in values:
            raise ValueError(f"Missing config section: [{section}]")
    seed = _integer("seed", "seed", values.get("seed", 640))

    sources = _sources(values["sources"])

    search = _table(
        values["search"],
        "search",
        {"hash_mb", "timeout_seconds", "joint"},
        {"syzygy_path", "syzygy_probe_limit"},
    )
    hash_mb = _integer("search", "hash_mb", search["hash_mb"], 1)
    timeout = _integer("search", "timeout_seconds", search["timeout_seconds"], 1)
    joint = _joint_distribution(search["joint"])
    syzygy_path = search.get("syzygy_path")
    if syzygy_path is not None and not isinstance(syzygy_path, str):
        raise ValueError("[search] syzygy_path must be a path string")
    syzygy_path = syzygy_path or None
    probe_limit = _integer(
        "search", "syzygy_probe_limit", search.get("syzygy_probe_limit", 7), 1
    )
    if probe_limit > 7:
        raise ValueError("[search] syzygy_probe_limit must be at most 7")

    synthesis = _table(
        values["synthesis"],
        "synthesis",
        {
            "claim_draw",
            "board_descent_hazard",
            "board_frontier_p",
            "presentations",
            "random_min_fraction",
            "random_max_fraction",
        },
    )
    if not isinstance(synthesis["claim_draw"], bool):
        raise ValueError("[synthesis] claim_draw must be a boolean")
    presentations = _distribution(
        "synthesis", "presentations", synthesis["presentations"], _presentation_key
    )
    minimum = _probability(
        "synthesis", "random_min_fraction", synthesis["random_min_fraction"]
    )
    maximum = _probability(
        "synthesis", "random_max_fraction", synthesis["random_max_fraction"]
    )
    if minimum >= maximum:
        raise ValueError("[synthesis] random cut interval must be nonempty")
    synthesis = SynthesisConfig(
        synthesis["claim_draw"],
        _probability(
            "synthesis", "board_descent_hazard", synthesis["board_descent_hazard"]
        ),
        _probability("synthesis", "board_frontier_p", synthesis["board_frontier_p"]),
        minimum,
        maximum,
    )

    release = _table(
        values["release"],
        "release",
        {"shard_size", "workers", "start_shard"},
        {"end_shard"},
    )
    shard_size = _integer("release", "shard_size", release["shard_size"], 1)
    workers = _integer("release", "workers", release["workers"], 1)
    start_shard = _integer("release", "start_shard", release["start_shard"], 0)
    end_shard = release.get("end_shard")
    if end_shard is not None:
        end_shard = _integer("release", "end_shard", end_shard, start_shard + 1)
    release = ReleaseConfig(shard_size, workers, start_shard, end_shard)
    search = SearchConfig(hash_mb, timeout, syzygy_path, probe_limit)
    sampling = SamplingConfig(joint, presentations)

    merge = _table(
        values["merge"],
        "merge",
        {
            "vocab_size",
            "fit_split",
            "holdout_fraction",
            "min_count",
            "per_length_limit",
            "selection_stride",
            "max_rounds",
            "top",
            "batch_size",
        },
        {"max_roots", "block", "workers"},
    )
    fit_split = merge["fit_split"]
    if fit_split not in {"all", "train", "val"}:
        raise ValueError("[merge] fit_split must be all, train, or val")
    holdout = _probability("merge", "holdout_fraction", merge["holdout_fraction"])
    if holdout == 1:
        raise ValueError("[merge] holdout_fraction must be less than one")
    merge = MergeConfig(
        _integer("merge", "vocab_size", merge["vocab_size"], 1),
        fit_split,
        holdout,
        _integer("merge", "max_roots", merge["max_roots"], 1)
        if "max_roots" in merge
        else None,
        _integer("merge", "min_count", merge["min_count"], 1),
        _integer("merge", "per_length_limit", merge["per_length_limit"], 1),
        _integer("merge", "selection_stride", merge["selection_stride"], 1),
        _integer("merge", "block", merge["block"], 1) if "block" in merge else None,
        _integer("merge", "max_rounds", merge["max_rounds"], 1),
        _integer("merge", "top", merge["top"], 0),
        _integer("merge", "batch_size", merge["batch_size"], 1),
        _integer("merge", "workers", merge.get("workers", 1), 1),
    )
    return DataConfig(seed, sources, search, synthesis, release, sampling, merge)
