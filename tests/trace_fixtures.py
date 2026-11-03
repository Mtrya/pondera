"""Constructed edge-case records derived from the real trace fixtures."""

import json
import re
from pathlib import Path


def trace_config_text(width=1, depth=4, joint=None, presentations=None, engine=None):
    text = Path("configs/trace.toml").read_text(encoding="utf-8")
    if joint is None:
        joint = [{"width": width, "depth": depth, "probability": 1.0}]
    entries = [
        "{ "
        + ", ".join(f"{key} = {json.dumps(value)}" for key, value in entry.items())
        + " }"
        for entry in joint
    ]
    search = "[search]\nhash_mb = 256\njoint = [" + ", ".join(entries) + "]\n\n"
    text = re.sub(r"(?ms)^\[search\]\n.*?(?=^\[)", lambda _: search, text)
    if presentations is not None:
        table = (
            "{ "
            + ", ".join(f"{key} = {value}" for key, value in presentations.items())
            + " }"
        )
        text = re.sub(
            r"^presentations =.*$",
            lambda _: f"presentations = {table}",
            text,
            flags=re.MULTILINE,
        )
    if engine is not None:
        text = re.sub(
            r"^engine =.*$",
            lambda _: f"engine = {json.dumps(str(engine))}",
            text,
            flags=re.MULTILINE,
        )
    return text


def tie_sidecar_lines(case, lines):
    """Make round one of middlegame_black_w3 end on an exact value tie.

    The real round ranks f6e4 (-2360) over c5f2 (-2458). The copy gives the
    rank to c5f2 and makes the values equal, so the written values alone
    cannot name the champion. It is only used with that fixture.
    """
    if case != "middlegame_black_w3":
        return lines
    events = [json.loads(line) for line in lines]

    def event(**fields):
        matches = [
            item
            for item in events
            if all(item.get(key) == value for key, value in fields.items())
        ]
        if len(matches) != 1:
            raise AssertionError(f"expected one event matching {fields}")
        return matches[0]

    promote = event(type="candidate", round=1, attempt=2, call="promote", move="c5f2")
    promote["raw"], promote["cp"] = 2360, 624
    frontier = event(type="frontier", round=1, attempt=2, path=["c5f2"])
    frontier["raw"], frontier["cp"] = 2360, 624
    attempt = event(type="attempt", round=1, attempt=2)
    attempt["raw"], attempt["cp"] = -2360, -624
    end = event(type="round_end", round=1)
    rank = {entry["move"]: entry for entry in end["rank"]}
    rank["c5f2"].update(raw=-2360, cp=-624)
    end["rank"] = [rank["c5f2"], rank["f6e4"], rank["c6d4"]]
    return [json.dumps(item, separators=(",", ":")) for item in events]


def read_release_rows(directory):
    from pondera.data.pipeline import load_manifest, verify_shard_files

    directory = Path(directory)
    release = load_manifest(directory / "release.json")
    output = {}
    for shard in release["shards"]:
        _, rows, arrays = verify_shard_files(directory, shard)
        for index, row in enumerate(rows):
            length = int(arrays["trace_len"][index])
            count = int(arrays["state_count"][index])
            output[(row["root_id"], row["presentation"])] = {
                "row": row,
                "trace_tokens": tuple(
                    int(value) for value in arrays["trace_tokens"][index, :length]
                ),
                "trace_roles": tuple(
                    int(value) for value in arrays["trace_roles"][index, :length]
                ),
                "state": {
                    name: tuple(int(value) for value in arrays[name][index, :count])
                    for name in (
                        "state_best",
                        "state_value",
                        "state_basis",
                        "state_without_board",
                        "state_with_board",
                    )
                },
            }
    return output
