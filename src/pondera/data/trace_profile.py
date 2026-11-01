"""Streaming statistics over synthesized grammar-v2 traces."""

import statistics
from collections import Counter

import numpy as np

from pondera.data.trace_synthesis import PRESENTATIONS


def _percentiles(values):
    if not values:
        return {"count": 0, "min": 0, "p10": 0.0, "median": 0.0, "p90": 0.0, "max": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": int(min(values)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": int(max(values)),
        "mean": float(statistics.mean(values)),
    }


class RootProfile:
    """Streaming aggregate of synthesized-root statistics.

    Roots are consumed one at a time, so a release can compute its statistics
    without retaining the token arrays of every completed shard.
    """

    def __init__(self, presentations):
        self.presentations = presentations
        self.roots = 0
        self.rows = 0
        self.transitions = 0
        self.overturns = 0
        self.refutations = 0
        self.rounds = 0
        self.full_lengths = []
        self.total_lengths = []
        self.presentation_lengths = {name: [] for name in PRESENTATIONS}
        self.presentation_thinking = {name: [] for name in PRESENTATIONS}
        self.presentation_counts = Counter()
        self.rounds_distribution = Counter()
        self.converged_distribution = Counter()
        self.refutation_distribution = Counter()
        self.value_shapes = Counter()
        self.splits = Counter()
        self.roots_with_overturn = 0
        self.first_round_matches_final = 0
        self.all_draw_roots = 0
        self.claim_draw_roots = 0
        self.claimable_roots = 0
        self.board_descent = 0
        self.board_frontier = 0
        self.exact_evals = 0
        self.terminal_evals = 0
        self.frontier_evals = 0
        self.omitted_bound = 0
        self.omitted_early_refutation = 0
        self.omitted_root_fail_high = 0
        self.per_root = []

    def add(self, root):
        """Fold one synthesized root into the aggregate."""
        stats = root.stats
        selected = self.presentations[root.root["root_id"]]
        self.roots += 1
        self.rows += len(selected)
        self.transitions += max(stats["rounds"] - 1, 0)
        self.overturns += stats["overturned_rounds"]
        self.refutations += stats["refutations"]
        self.rounds += stats["rounds"]
        self.full_lengths.append(stats["thinking_tokens"])
        self.total_lengths.append(stats["total_tokens"])
        for name in selected:
            row = root.rows[name]
            self.presentation_lengths[name].append(len(row.tokens))
            self.presentation_thinking[name].append(row.thinking_tokens)
            self.presentation_counts[name] += 1
        self.rounds_distribution[stats["rounds"]] += 1
        self.converged_distribution[stats["converged_depth"]] += 1
        for count in stats["refutations_per_round"]:
            self.refutation_distribution[count] += 1
        values = [value for value in stats["root_values"] if value is not None]
        if not values:
            self.value_shapes["mate_only"] += 1
        else:
            changes = sum(1 for left, right in zip(values, values[1:]) if left != right)
            if changes == 0:
                self.value_shapes["flat"] += 1
            elif all(right >= left for left, right in zip(values, values[1:])):
                self.value_shapes["monotone_up"] += 1
            elif all(right <= left for left, right in zip(values, values[1:])):
                self.value_shapes["monotone_down"] += 1
            else:
                self.value_shapes["mixed"] += 1
        self.splits[root.root["split"]] += 1
        self.roots_with_overturn += stats["overturned_rounds"] > 0
        self.first_round_matches_final += stats["first_round_matches_final"]
        self.all_draw_roots += stats["all_draw"]
        self.claim_draw_roots += stats["claim_draw"]
        self.claimable_roots += stats["claimable"]
        self.board_descent += stats["board_descent"]
        self.board_frontier += stats["board_frontier"]
        self.exact_evals += stats["exact_evals"]
        self.terminal_evals += stats["terminal_evals"]
        self.frontier_evals += stats["frontier_evals"]
        self.omitted_bound += stats["omitted_bound"]
        self.omitted_early_refutation += stats["omitted_early_refutation"]
        self.omitted_root_fail_high += stats["omitted_root_fail_high"]
        self.per_root.append(
            {
                "root_id": root.root["root_id"],
                "split": root.root["split"],
                "rounds": stats["rounds"],
                "thinking_tokens": stats["thinking_tokens"],
                "overturned_rounds": stats["overturned_rounds"],
                "all_draw": stats["all_draw"],
                "claim_draw": stats["claim_draw"],
            }
        )

    def result(self):
        """The statistics document consumed by release manifests."""
        if not self.roots:
            raise ValueError("A statistics profile needs at least one root")
        return {
            "roots": self.roots,
            "rows": self.rows,
            "split_counts": dict(sorted(self.splits.items())),
            "presentation_counts": {
                name: self.presentation_counts[name] for name in PRESENTATIONS
            },
            "rounds": dict(sorted(self.rounds_distribution.items())),
            "rounds_summary": _percentiles([root["rounds"] for root in self.per_root]),
            "thinking_tokens": _percentiles(self.full_lengths),
            "total_tokens": _percentiles(self.total_lengths),
            "presentation_lengths": {
                name: _percentiles(values)
                for name, values in self.presentation_lengths.items()
            },
            "presentation_thinking_tokens": {
                name: _percentiles(values)
                for name, values in self.presentation_thinking.items()
            },
            "overturned_rounds": self.overturns,
            "overturn_rate": self.overturns / self.transitions
            if self.transitions
            else 0.0,
            "roots_with_overturn": self.roots_with_overturn,
            "refutations": self.refutations,
            "refutations_per_round": self.refutations / max(self.rounds, 1),
            "refutations_per_round_distribution": dict(
                sorted(self.refutation_distribution.items())
            ),
            "converged_depth": dict(sorted(self.converged_distribution.items())),
            "first_round_matches_final": self.first_round_matches_final,
            "value_trajectory_shapes": dict(sorted(self.value_shapes.items())),
            "all_draw_roots": self.all_draw_roots,
            "claim_draw_roots": self.claim_draw_roots,
            "claimable_roots": self.claimable_roots,
            "board_descent": self.board_descent,
            "board_frontier": self.board_frontier,
            "exact_evals": self.exact_evals,
            "terminal_evals": self.terminal_evals,
            "frontier_evals": self.frontier_evals,
            "omitted_bound": self.omitted_bound,
            "omitted_early_refutation": self.omitted_early_refutation,
            "omitted_root_fail_high": self.omitted_root_fail_high,
            "per_root": self.per_root,
        }


def profile_roots(roots, presentations=None):
    """Aggregate the statistics over synthesized roots and their written rows."""
    roots = list(roots)
    if not roots:
        raise ValueError("A statistics profile needs at least one root")
    if presentations is None:
        presentations = {root.root["root_id"]: PRESENTATIONS for root in roots}
    profile = RootProfile(presentations)
    for root in roots:
        profile.add(root)
    return profile.result()
