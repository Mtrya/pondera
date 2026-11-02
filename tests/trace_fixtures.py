"""Constructed edge-case records derived from the real trace fixtures."""

import json


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
