"""Validated TOML configuration for trace-v2 collection and synthesis."""

import math
import random
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("configs/trace.toml")
DEFAULT_TEACHER_PATH = Path("docs/data.md")

_SECTIONS = {
    "engine": {"path"},
    "search": {
        "depth",
        "teacher_width",
        "threads",
        "hash_mb",
        "fresh_engine_per_root",
        "workers",
    },
    "board": {"board_descent_hazard", "board_frontier_p"},
    "presentation": {
        "zero",
        "full",
        "random",
        "last_root",
        "random_min_fraction",
        "random_max_fraction",
    },
    "synthesis": {"confirmatory_repetition", "claim_draw"},
    "smoke": {"roots", "seed", "output"},
    "recipe": {"widths", "depths", "presentations"},
}
_OPTIONAL_SECTIONS = {"recipe"}
_RECIPE_PRESENTATIONS = ("zero", "full", "random")
_DIGITS = re.compile(r"^[1-9][0-9]*$")


def _reject_unknown(section, values):
    unknown = set(values) - _SECTIONS[section]
    if unknown:
        raise ValueError(f"Unknown config key in [{section}]: {sorted(unknown)[0]}")
    missing = _SECTIONS[section] - set(values)
    if missing:
        raise ValueError(f"Missing config key in [{section}]: {sorted(missing)[0]}")


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


@dataclass(frozen=True)
class EngineConfig:
    path: str


@dataclass(frozen=True)
class SearchConfig:
    depth: int
    teacher_width: int
    threads: int
    hash_mb: int
    fresh_engine_per_root: bool
    workers: int


@dataclass(frozen=True)
class BoardConfig:
    board_descent_hazard: float
    board_frontier_p: float


@dataclass(frozen=True)
class PresentationConfig:
    zero: float
    full: float
    random: float
    last_root: float
    random_min_fraction: float
    random_max_fraction: float

    def fractions(self):
        return {
            "zero": self.zero,
            "full": self.full,
            "random": self.random,
            "last_root": self.last_root,
        }


@dataclass(frozen=True)
class RecipeConfig:
    """Optional per-root distributions over teacher width, depth, presentation."""

    widths: tuple = None
    depths: tuple = None
    presentations: tuple = None

    def identity(self):
        payload = {}
        for name, distribution in (
            ("widths", self.widths),
            ("depths", self.depths),
            ("presentations", self.presentations),
        ):
            if distribution is not None:
                payload[name] = {
                    str(key): probability for key, probability in distribution
                }
        return payload

    def widest(self, default_width):
        return max((key for key, _ in self.widths), default=default_width)

    def samples_presentations(self):
        return self.presentations is not None


def _recipe_key(name, value):
    if not isinstance(value, str) or not _DIGITS.match(value):
        raise ValueError(f"[recipe] {name} keys must be positive integers")
    return int(value)


def _presentation_key(name, value):
    if value not in _RECIPE_PRESENTATIONS:
        raise ValueError(f"[recipe] {name} keys must be one of {_RECIPE_PRESENTATIONS}")
    return value


def _recipe_distribution(name, values, key_parser):
    if not isinstance(values, dict) or not values:
        raise ValueError(f"[recipe] {name} must be a nonempty table")
    distribution = {}
    for raw_key, probability in values.items():
        key = key_parser(name, raw_key)
        if key in distribution:
            raise ValueError(f"[recipe] {name} repeats a key")
        distribution[key] = _probability("recipe", f"{name}.{raw_key}", probability)
    total = sum(distribution.values())
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError(f"[recipe] {name} probabilities must sum to 1")
    return tuple(sorted(distribution.items()))


def parse_recipe(values):
    """Validate one recipe table, whether from TOML or a release manifest."""
    if not isinstance(values, dict):
        raise ValueError("[recipe] must be a table")
    unknown = set(values) - _SECTIONS["recipe"]
    if unknown:
        raise ValueError(f"Unknown config key in [recipe]: {sorted(unknown)[0]}")
    if not values:
        raise ValueError("[recipe] must define at least one distribution")
    return RecipeConfig(
        widths=_recipe_distribution("widths", values["widths"], _recipe_key)
        if "widths" in values
        else None,
        depths=_recipe_distribution("depths", values["depths"], _recipe_key)
        if "depths" in values
        else None,
        presentations=_recipe_distribution(
            "presentations", values["presentations"], _presentation_key
        )
        if "presentations" in values
        else None,
    )


def sample_distribution(rng, distribution):
    """Draw one key from a frozen sorted distribution."""
    roll = rng.random()
    cumulative = 0.0
    fallback = distribution[0][0]
    for key, probability in distribution:
        if probability > 0.0:
            fallback = key
        cumulative += probability
        if roll < cumulative:
            return key
    return fallback


def sample_width(recipe, seed, root_id, default_width):
    """Sample the configured width for one root, independent of other tables."""
    if recipe is None or recipe.widths is None:
        return default_width
    return sample_distribution(random.Random(f"{seed}:{root_id}:width"), recipe.widths)


def sample_depth(recipe, seed, root_id, default_depth):
    """Sample the search depth for one root, independent of other tables."""
    if recipe is None or recipe.depths is None:
        return default_depth
    return sample_distribution(random.Random(f"{seed}:{root_id}:depth"), recipe.depths)


def sample_presentation(recipe, seed, root_id):
    """Sample the release presentation for one root, or None for all four."""
    if recipe is None or recipe.presentations is None:
        return None
    return sample_distribution(
        random.Random(f"{seed}:{root_id}:presentation"), recipe.presentations
    )


@dataclass(frozen=True)
class SynthesisConfig:
    confirmatory_repetition: str
    claim_draw: bool


@dataclass(frozen=True)
class SmokeConfig:
    roots: int
    seed: int
    output: str


@dataclass(frozen=True)
class TraceConfig:
    engine: EngineConfig
    search: SearchConfig
    board: BoardConfig
    presentation: PresentationConfig
    synthesis: SynthesisConfig
    smoke: SmokeConfig
    source_path: Path
    recipe: RecipeConfig = None

    def search_dict(self):
        return {
            "depth": self.search.depth,
            "teacher_width": self.search.teacher_width,
            "threads": self.search.threads,
            "hash_mb": self.search.hash_mb,
            "fresh_engine_per_root": self.search.fresh_engine_per_root,
        }

    def board_dict(self):
        return {
            "board_descent_hazard": self.board.board_descent_hazard,
            "board_frontier_p": self.board.board_frontier_p,
        }

    def presentation_dict(self):
        return {
            "zero": self.presentation.zero,
            "full": self.presentation.full,
            "random": self.presentation.random,
            "last_root": self.presentation.last_root,
            "random_min_fraction": self.presentation.random_min_fraction,
            "random_max_fraction": self.presentation.random_max_fraction,
        }

    def synthesis_dict(self):
        return {
            "confirmatory_repetition": self.synthesis.confirmatory_repetition,
            "claim_draw": self.synthesis.claim_draw,
        }


def load_config(path=DEFAULT_CONFIG_PATH):
    """Load and validate the single trace-v2 knob file."""
    path = Path(path)
    with path.open("rb") as handle:
        values = tomllib.load(handle)
    unknown = set(values) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"Unknown config section: {sorted(unknown)[0]}")
    for section in set(_SECTIONS) - _OPTIONAL_SECTIONS:
        if section not in values:
            raise ValueError(f"Missing config section: [{section}]")
        if not isinstance(values[section], dict):
            raise ValueError(f"Config section [{section}] must be a table")
        _reject_unknown(section, values[section])
    if "recipe" in values and not isinstance(values["recipe"], dict):
        raise ValueError("Config section [recipe] must be a table")

    engine = values["engine"]
    if not isinstance(engine["path"], str) or not engine["path"]:
        raise ValueError("[engine] path must be a nonempty string")
    search = values["search"]
    depth = _integer("search", "depth", search["depth"], 1)
    width = _integer("search", "teacher_width", search["teacher_width"], 1)
    threads = _integer("search", "threads", search["threads"], 1)
    hash_mb = _integer("search", "hash_mb", search["hash_mb"], 1)
    workers = _integer("search", "workers", search["workers"], 1)
    fresh = search["fresh_engine_per_root"]
    if not isinstance(fresh, bool):
        raise ValueError("[search] fresh_engine_per_root must be a boolean")
    if threads != 1:
        raise ValueError("threads must be 1 for a reproducible search record")
    if not fresh:
        raise ValueError("fresh_engine_per_root must be true for a release")

    board = values["board"]
    hazard = _probability(
        "board", "board_descent_hazard", board["board_descent_hazard"]
    )
    frontier_p = _probability("board", "board_frontier_p", board["board_frontier_p"])

    presentation = values["presentation"]
    fractions = {
        name: _probability("presentation", name, presentation[name])
        for name in ("zero", "full", "random", "last_root")
    }
    total = sum(fractions.values())
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError("[presentation] fractions must sum to 1")
    random_min = _probability(
        "presentation", "random_min_fraction", presentation["random_min_fraction"]
    )
    random_max = _probability(
        "presentation", "random_max_fraction", presentation["random_max_fraction"]
    )
    if not random_min < random_max:
        raise ValueError("[presentation] random cut interval must be nonempty")

    synthesis = values["synthesis"]
    if synthesis["confirmatory_repetition"] != "faithful":
        raise ValueError('confirmatory_repetition must be "faithful"')
    if not isinstance(synthesis["claim_draw"], bool):
        raise ValueError("[synthesis] claim_draw must be a boolean")

    smoke = values["smoke"]
    roots = _integer("smoke", "roots", smoke["roots"], 1)
    seed = _integer("smoke", "seed", smoke["seed"])
    if not isinstance(smoke["output"], str) or not smoke["output"]:
        raise ValueError("[smoke] output must be a nonempty string")

    recipe = parse_recipe(values["recipe"]) if "recipe" in values else None
    return TraceConfig(
        engine=EngineConfig(engine["path"]),
        search=SearchConfig(depth, width, threads, hash_mb, fresh, workers),
        board=BoardConfig(hazard, frontier_p),
        presentation=PresentationConfig(
            fractions["zero"],
            fractions["full"],
            fractions["random"],
            fractions["last_root"],
            random_min,
            random_max,
        ),
        synthesis=SynthesisConfig(
            synthesis["confirmatory_repetition"], synthesis["claim_draw"]
        ),
        smoke=SmokeConfig(roots, seed, smoke["output"]),
        source_path=path,
        recipe=recipe,
    )
