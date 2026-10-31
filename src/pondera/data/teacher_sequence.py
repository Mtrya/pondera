"""Teacher search trees and trace programs for sequence data synthesis.

Fixed teachers walk a bounded tree depth first. The adaptive teacher first
screens every root candidate with d1 and d8, then deepens unstable candidates
in complete rounds at target depths two and three. Every non-terminal node is
annotated once with Stockfish and shares that annotation across teachers. Node
identity is the absolute UCI path, so a transposition reached by two paths is
two nodes; revisiting an existing path does not spend another node.

Annotations are read from the node side to move and converted to the root
player's frame. Operation logs, truncation states, parent updates, and final
decisions all use the same quantized value and mate semantics as the token
vocabulary. Raw engine values are used only to guide teacher expansion. Parent
updates prefer root-player wins and shorter forced mates, delay forced losses,
and never treat a partial child set as a complete subtree. Mate-in-one closes a
node immediately after its d1 and d8 baseline.

The operation log is the single semantic source for navigation, evaluations,
root candidate state, and the final decision. ``emit_operations`` turns it into
no-tool and BOARD-augmented traces, and ``verify_teacher_trace`` independently
checks their rule states, numerical updates, and complete child sets.
"""

import time
from dataclasses import dataclass
from typing import Callable, Mapping

import chess
import chess.engine

from pondera.chess.codec import decode_value_token, mate_token, value_bin
from pondera.chess.tracker import PathTracker
from pondera.chess.vocabulary import VALUE
from pondera.data.annotation import AnnotationConfig, analyse
from pondera.data.trace import TraceWriter, replay

REQUIRED_DEPTHS = (1, 4, 8)

# Adaptive expansion: the shallow stability rule from the frozen design.
STABILITY_VALUE_FLOOR = 0.05
STABILITY_NEAR_VALUE = 0.10
MAX_ADAPTIVE_CANDIDATES = 4

# WDL values are per-mille ratios, so a difference of exactly 0.05 can land a
# few ulps below the literal. Thresholds are inclusive by intent.
VALUE_EPSILON = 1e-9

# BOARD is requested before a non-root expansion when the node was left at
# least eight plies ago, or when sixteen plies have advanced without an
# observation. The root board prefix is always observed, so the root never
# requests BOARD; a request is never repeated without a new move in between.
BOARD_RETURN_PLIES = 8
BOARD_PLY_INTERVAL = 16


class TeacherError(ValueError):
    """A teacher tree, annotation, or trace violated its contract."""


class IncompleteAnalysis(TeacherError):
    """A node did not produce complete d1/d4/d8 rank snapshots."""


@dataclass(frozen=True)
class Candidate:
    """One ranked engine candidate at one completed depth."""

    rank: int
    move: str
    cp: int | None
    mate: int | None
    value: float
    wdl: tuple[int, int, int]
    pv: tuple[str, ...]

    @classmethod
    def from_entry(cls, rank: int, entry: Mapping) -> "Candidate":
        return cls(
            rank=rank,
            move=entry["bm"],
            cp=entry.get("cp"),
            mate=entry.get("mate"),
            value=float(entry["value"]),
            wdl=tuple(entry.get("wdl") or ()),
            pv=tuple(entry.get("pv") or ()),
        )

    def entry(self) -> dict:
        return {
            "bm": self.move,
            "cp": self.cp,
            "mate": self.mate,
            "pv": list(self.pv),
            "wdl": list(self.wdl),
            "value": self.value,
        }


@dataclass(frozen=True)
class NodeAnalysis:
    """A node's rule state plus complete d1/d4/d8 annotations.

    ``candidates`` values are exactly as the engine reported them, from the
    node side to move. Use ``root_value`` and ``root_mate`` to read them in the
    root player's frame.
    """

    path: tuple[str, ...]
    fen: str
    repetition: int
    turn: bool
    terminal: bool
    rule_value: float | None
    rule_mate: int | None
    rule_mate_win: bool | None
    rule_reason: str | None
    candidates: Mapping[int, tuple[Candidate, ...]]
    legal_moves: tuple[str, ...]
    nodes: int | None
    elapsed_ms: int
    reached_depth: int | None
    reported_depth: int | None
    incomplete_depths: tuple[int, ...]

    def rank1(self, depth: int) -> Candidate:
        try:
            return self.candidates[depth][0]
        except (KeyError, IndexError) as error:
            raise TeacherError(f"Node has no d{depth} rank 1: {self.path}") from error

    def to_record(self) -> dict:
        record = {
            "path": list(self.path),
            "fen": self.fen,
            "repetition": self.repetition,
            "turn": "white" if self.turn else "black",
            "terminal": self.terminal,
            "legal_moves": list(self.legal_moves),
            "per_depth": [
                {
                    "d": depth,
                    **{
                        f"r{rank}": row[rank - 1].entry()
                        for rank in range(1, len(row) + 1)
                    },
                }
                for depth, row in sorted(self.candidates.items())
            ],
            "nodes": self.nodes,
            "elapsed_ms": self.elapsed_ms,
            "reached_depth": self.reached_depth,
            "reported_depth": self.reported_depth,
            "incomplete_depths": list(self.incomplete_depths),
        }
        if self.terminal:
            record["rule"] = {
                "value": self.rule_value,
                "mate": self.rule_mate,
                "mate_win": self.rule_mate_win,
                "reason": self.rule_reason,
            }
        return record

    @classmethod
    def from_record(cls, record: Mapping) -> "NodeAnalysis":
        candidates = {}
        for row in record.get("per_depth", []):
            depth = row["d"]
            ranks = sorted(int(name[1:]) for name in row if name.startswith("r"))
            candidates[depth] = tuple(
                Candidate.from_entry(rank, row[f"r{rank}"]) for rank in ranks
            )
        rule = record.get("rule") or {}
        return cls(
            path=tuple(record["path"]),
            fen=record["fen"],
            repetition=record["repetition"],
            turn=record["turn"] == "white",
            terminal=bool(record["terminal"]),
            rule_value=rule.get("value"),
            rule_mate=rule.get("mate"),
            rule_mate_win=rule.get("mate_win"),
            rule_reason=rule.get("reason"),
            candidates=candidates,
            legal_moves=tuple(record.get("legal_moves", [])),
            nodes=record.get("nodes"),
            elapsed_ms=record.get("elapsed_ms", 0),
            reached_depth=record.get("reached_depth"),
            reported_depth=record.get("reported_depth"),
            incomplete_depths=tuple(record.get("incomplete_depths", [])),
        )


def root_value(node: NodeAnalysis, depth: int, rank: int = 1) -> float:
    """A candidate value read from the root player's side to move.

    ``depth`` selects the engine snapshot (1, 4, or 8); the perspective flip
    depends on the node's path depth, not on how deep the search was.
    """
    value = node.candidates[depth][rank - 1].value
    return value if len(node.path) % 2 == 0 else -value


def root_mate(node: NodeAnalysis, depth: int, rank: int = 1):
    """A candidate mate as (distance, root player delivers it)."""
    mate = node.candidates[depth][rank - 1].mate
    if mate is None:
        return None, None
    return abs(mate), (mate > 0) == (len(node.path) % 2 == 0)


def _terminal_rule(board: chess.Board, root_turn: bool) -> tuple:
    if board.is_checkmate():
        root_wins = board.turn != root_turn
        return (
            "checkmate",
            1.0 if root_wins else -1.0,
            0,
            root_wins,
        )
    return ("draw", 0.0, None, None)


def analyse_node(
    engine,
    tracker: PathTracker,
    config: AnnotationConfig,
    root_turn: bool,
) -> NodeAnalysis:
    """Annotate the tracker's current node, or apply the terminal rule."""
    board = tracker.board
    repetition = tracker.repetition()
    fen = tracker.fen()
    legal = tuple(sorted(move.uci() for move in board.legal_moves))
    if board.is_game_over() or repetition >= 5:
        reason, value, mate, mate_win = _terminal_rule(board, root_turn)
        return NodeAnalysis(
            path=tracker.path,
            fen=fen,
            repetition=repetition,
            turn=board.turn,
            terminal=True,
            rule_value=value,
            rule_mate=mate,
            rule_mate_win=mate_win,
            rule_reason=reason,
            candidates={},
            legal_moves=legal,
            nodes=None,
            elapsed_ms=0,
            reached_depth=None,
            reported_depth=None,
            incomplete_depths=(),
        )
    start = time.monotonic()
    unit = analyse(engine, board.copy(), config)
    elapsed_ms = int((time.monotonic() - start) * 1000)
    if not unit["requested_depth_completed"]:
        raise IncompleteAnalysis(
            f"Node {tracker.path} stopped at depth {unit['reached_depth']}; "
            f"incomplete {unit['incomplete_depths']}"
        )
    expected = unit["expected_candidates"]
    if expected < 1:
        raise IncompleteAnalysis(f"Node {tracker.path} has no engine candidates")
    by_depth = {row["d"]: row for row in unit["per_depth"]}
    legal_set = set(legal)
    candidates = {}
    for depth in REQUIRED_DEPTHS:
        row = by_depth.get(depth)
        if row is None:
            raise IncompleteAnalysis(
                f"Node {tracker.path} is missing d{depth}; "
                f"reached {unit['reached_depth']}"
            )
        ranks = {f"r{rank}" for rank in range(1, expected + 1)}
        if not ranks.issubset(row):
            raise IncompleteAnalysis(
                f"Node {tracker.path} d{depth} lacks ranks {sorted(ranks - set(row))}"
            )
        entries = []
        for rank in range(1, expected + 1):
            entry = row[f"r{rank}"]
            if entry["bm"] not in legal_set:
                raise IncompleteAnalysis(
                    f"Node {tracker.path} d{depth} rank {rank} is not legal: "
                    f"{entry['bm']}"
                )
            entries.append(Candidate.from_entry(rank, entry))
        candidates[depth] = tuple(entries)
    return NodeAnalysis(
        path=tracker.path,
        fen=fen,
        repetition=repetition,
        turn=board.turn,
        terminal=False,
        rule_value=None,
        rule_mate=None,
        rule_mate_win=None,
        rule_reason=None,
        candidates=candidates,
        legal_moves=legal,
        nodes=unit["nodes"],
        elapsed_ms=elapsed_ms,
        reached_depth=unit["reached_depth"],
        reported_depth=unit["reported_depth"],
        incomplete_depths=tuple(unit["incomplete_depths"]),
    )


class AnalysisCache:
    """One engine per worker, one annotation per absolute path."""

    def __init__(self, engine, config: AnnotationConfig, root_turn: bool):
        self.engine = engine
        self.config = config
        self.root_turn = root_turn
        self._nodes = {}
        self.engine_calls = 0
        self.analysis_ms = 0

    def __call__(self, tracker: PathTracker) -> NodeAnalysis:
        key = tracker.path
        node = self._nodes.get(key)
        if node is None:
            node = analyse_node(self.engine, tracker, self.config, self.root_turn)
            if node.path != key:
                raise TeacherError("Annotation path does not match its tracker")
            self._nodes[key] = node
            if not node.terminal:
                self.engine_calls += 1
                self.analysis_ms += node.elapsed_ms
        return node

    def records(self) -> list[dict]:
        ordered = sorted(self._nodes, key=lambda path: (len(path), path))
        return [self._nodes[path].to_record() for path in ordered]

    def nodes_by_path(self) -> dict:
        return dict(self._nodes)

    def __len__(self):
        return len(self._nodes)


@dataclass(frozen=True)
class TeacherSpec:
    """One teacher configuration; identity includes every field."""

    name: str
    kind: str
    max_depth: int
    max_nodes: int
    width: int = 2
    value_floor: float = STABILITY_VALUE_FLOOR
    near_value: float = STABILITY_NEAR_VALUE
    max_candidates: int = MAX_ADAPTIVE_CANDIDATES

    def __post_init__(self):
        if self.kind not in {"fixed", "adaptive"}:
            raise ValueError(f"Unknown teacher kind: {self.kind}")
        if min(self.max_depth, self.max_nodes, self.width, self.max_candidates) < 1:
            raise ValueError("Teacher limits must be positive")
        if not 0 < self.value_floor <= 1 or not 0 <= self.near_value <= 2:
            raise ValueError("Teacher value thresholds are out of range")

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
            "width": self.width,
            "value_floor": self.value_floor,
            "near_value": self.near_value,
            "max_candidates": self.max_candidates,
        }


TEACHER_SPECS = (
    TeacherSpec("fixed-depth-3-width-2", "fixed", max_depth=3, max_nodes=15),
    TeacherSpec("fixed-depth-4-width-2", "fixed", max_depth=4, max_nodes=31),
    TeacherSpec("shallow-stability", "adaptive", max_depth=3, max_nodes=61),
)
TEACHER_BY_NAME = {spec.name: spec for spec in TEACHER_SPECS}
DEFAULT_TEACHER_SPECS = (
    TEACHER_BY_NAME["fixed-depth-3-width-2"],
    TEACHER_BY_NAME["shallow-stability"],
)


@dataclass(frozen=True)
class TreeResult:
    """The operation log of one teacher tree and its final decision."""

    spec: TeacherSpec
    operations: tuple[dict, ...]
    nodes: tuple[tuple[str, ...], ...]
    expanded: tuple[tuple[str, ...], ...]
    final_move: str
    final_basis: str
    final_value: float | None


def _entry(move, value, mate=None, mate_win=None, raw_value=None):
    return {
        "move": move,
        "value": value,
        "mate": mate,
        "mate_win": mate_win,
        **({"raw_value": raw_value} if raw_value is not None else {}),
    }


def _visible_entry(move, value, mate, mate_win):
    token = mate_token(mate, mate_win) if mate is not None else VALUE + value_bin(value)
    return _entry(move, decode_value_token(token), mate, mate_win, value)


def _prefer(left, right) -> float:
    if left["value"] != right["value"]:
        return left["value"] - right["value"]
    left_mate, right_mate = left["mate"], right["mate"]
    if left_mate is None and right_mate is None:
        return 0
    if left_mate is None:
        return -1 if right["mate_win"] else 1
    if right_mate is None:
        return 1 if left["mate_win"] else -1
    if left["mate_win"] != right["mate_win"]:
        return 1 if left["mate_win"] else -1
    if left["mate_win"]:
        return right_mate - left_mate
    return left_mate - right_mate


def _aggregate(parent_turn: bool, root_turn: bool, completed: list[dict]):
    if not completed:
        raise TeacherError("A parent update needs at least one completed child")
    chooser = max if parent_turn == root_turn else min
    chosen = chooser(completed, key=lambda entry: _prefer(entry, completed[0]))
    aggregate = dict(chosen)
    if aggregate["mate"] is not None:
        aggregate["mate"] += 1
    return aggregate, chosen["move"]


@dataclass(frozen=True)
class RootState:
    """The root candidate state after one operation boundary."""

    proposed: tuple[str, ...]
    completed: tuple[dict, ...]
    shallow: str | None
    shallow_value: float | None
    best: str | None
    basis: str | None
    value: float | None
    mate: int | None = None
    mate_win: bool | None = None


def _should_expand(spec, node, depth, node_count):
    if depth >= spec.max_depth or node_count >= spec.max_nodes:
        return False
    if spec.kind == "fixed":
        return True
    d1, d4, d8 = node.rank1(1), node.rank1(4), node.rank1(8)
    if len({d1.move, d4.move, d8.move}) > 1:
        return True
    v1, v4, v8 = root_value(node, 1), root_value(node, 4), root_value(node, 8)
    v = max(abs(v1 - v4), abs(v4 - v8))
    if v + VALUE_EPSILON >= spec.value_floor:
        return True
    candidates = node.candidates[8]
    if len(candidates) >= 2:
        gap = abs(root_value(node, 8, 1) - root_value(node, 8, 2))
        if gap <= max(spec.value_floor, v) + VALUE_EPSILON:
            return True
    return False


def _instability(node):
    v1, v4, v8 = root_value(node, 1), root_value(node, 4), root_value(node, 8)
    return max(abs(v1 - v4), abs(v4 - v8))


def _child_moves(spec, node):
    if spec.kind == "fixed":
        return tuple(candidate.move for candidate in node.candidates[8][: spec.width])
    moves = []
    limit = spec.max_candidates if not node.path else 2

    def add(move):
        if len(moves) < limit and move not in moves:
            moves.append(move)

    add(node.rank1(8).move)
    add(node.rank1(4).move)
    add(node.rank1(1).move)
    r1 = root_value(node, 8)
    for candidate in node.candidates[8]:
        if len(moves) >= limit:
            break
        if (
            abs(root_value(node, 8, candidate.rank) - r1)
            <= spec.near_value + VALUE_EPSILON
        ):
            add(candidate.move)
    return tuple(moves)


def _evaluation_fields(node, depth):
    mate, mate_win = root_mate(node, depth)
    fields = _visible_entry(None, root_value(node, depth), mate, mate_win)
    fields.pop("move")
    return {"d": depth, **fields, "rank1": node.rank1(depth).move}


def _best_state(proposed, completed, shallow, shallow_value, shallow_mate, shallow_win):
    if completed:
        chosen = max(completed, key=lambda entry: _prefer(entry, completed[0]))
        return (
            chosen["move"],
            "completed",
            chosen["value"],
            chosen["mate"],
            chosen["mate_win"],
        )
    if proposed:
        return proposed[0], "proposed", None, None, None
    if shallow is not None:
        return shallow, "shallow", shallow_value, shallow_mate, shallow_win
    return None, None, None, None, None


def derive_states(operations) -> list[RootState]:
    """Root candidate states after every non-final operation boundary."""
    proposed = []
    completed = []
    shallow = None
    shallow_value = None
    shallow_mate = None
    shallow_win = None
    states = []
    for operation in operations:
        kind = operation["op"]
        if kind == "final":
            break
        if kind == "move":
            path = tuple(operation["path"])
            if len(path) == 1 and operation["uci"] not in proposed:
                proposed.append(operation["uci"])
        elif kind == "evaluate":
            path = tuple(operation["path"])
            source = operation["source"]
            if path == ():
                if source in {"initial", "leaf"}:
                    shallow = operation["rank1"]
                    shallow_value = operation["value"]
                    shallow_mate = operation["mate"]
                    shallow_win = operation["mate_win"]
                elif source == "update":
                    completed = [dict(entry) for entry in operation["completed"]]
                elif source == "terminal":
                    raise TeacherError("A root position cannot be terminal")
            elif source not in {"initial", "leaf", "update", "terminal"}:
                raise TeacherError(f"Unknown evaluation source: {source}")
        elif kind not in {"back", "reset"}:
            raise TeacherError(f"Unknown operation: {kind}")
        best, basis, value, mate, mate_win = _best_state(
            proposed, completed, shallow, shallow_value, shallow_mate, shallow_win
        )
        states.append(
            RootState(
                proposed=tuple(proposed),
                completed=tuple(completed),
                shallow=shallow,
                shallow_value=shallow_value,
                best=best,
                basis=basis,
                value=value,
                mate=mate,
                mate_win=mate_win,
            )
        )
    return states


def _candidate_summary(state: RootState):
    values = {entry["move"]: entry["value"] for entry in state.completed}
    return [
        {
            "move": move,
            "order": order,
            "completed": move in values,
            "value": values.get(move),
        }
        for order, move in enumerate(state.proposed)
    ]


def simulate_tree(
    spec: TeacherSpec,
    root_fen: str,
    repetition: int,
    analyse_node_fn: Callable[[PathTracker], NodeAnalysis],
) -> TreeResult:
    """Grow one teacher tree with complete updates and visible decisions."""
    tracker = PathTracker(root_fen, repetition)
    root_node = analyse_node_fn(tracker)
    if not isinstance(root_node, NodeAnalysis):
        raise TeacherError("The analysis provider did not return a node")
    if root_node.path != tracker.path or root_node.terminal:
        raise TeacherError("The root annotation is invalid")
    root_turn = root_node.turn
    operations = []
    visited = {tracker.path}
    node_order = [tracker.path]
    expanded = []

    def terminal_operation(node):
        return {
            "op": "evaluate",
            "source": "terminal",
            "path": list(node.path),
            "value": node.rule_value,
            "mate": node.rule_mate,
            "mate_win": node.rule_mate_win,
            "terminal": node.rule_reason,
        }

    def evaluation_operation(node, depth, source):
        return {
            "op": "evaluate",
            "source": source,
            "path": list(node.path),
            **_evaluation_fields(node, depth),
        }

    def update_operation(node, completed):
        aggregate, chosen = _aggregate(node.turn, root_turn, completed)
        return {
            "op": "evaluate",
            "source": "update",
            "path": list(node.path),
            "value": aggregate["value"],
            "mate": aggregate["mate"],
            "mate_win": aggregate["mate_win"],
            "completed": [dict(entry) for entry in completed],
            "chosen": chosen,
        }

    def value_entry(node, depth):
        mate, mate_win = root_mate(node, depth)
        return _visible_entry(None, root_value(node, depth), mate, mate_win)

    def terminal_entry(node):
        return _entry(None, node.rule_value, node.rule_mate, node.rule_mate_win)

    def push(move):
        path = tracker.path + (move,)
        revisit = path in visited
        if not revisit and len(visited) >= spec.max_nodes:
            raise TeacherError("The node budget was exceeded before a complete update")
        operation = {"op": "move", "uci": move, "path": list(path)}
        if revisit:
            operation["revisit"] = True
        operations.append(operation)
        tracker.push(move)
        if not revisit:
            visited.add(path)
            node_order.append(path)
        return not revisit

    def mate_in_one():
        for move in tracker.board.legal_moves:
            board = tracker.board.copy()
            board.push(move)
            if board.is_checkmate():
                return move.uci()
        return None

    def visit(node, depth, first_visit=True):
        if node.path != tracker.path:
            raise TeacherError("The node annotation path does not match the tracker")
        if node.terminal:
            operations.append(terminal_operation(node))
            return terminal_entry(node)
        if first_visit:
            operations.append(evaluation_operation(node, 1, "initial"))
            operations.append(evaluation_operation(node, 8, "leaf"))
        mate = mate_in_one() if depth < spec.max_depth else None
        moves = (mate,) if mate is not None else ()
        if not moves and not _should_expand(spec, node, depth, len(visited)):
            return value_entry(node, 8)
        moves = moves or _child_moves(spec, node)
        if not moves:
            raise TeacherError(f"An expanding node has no candidates: {node.path}")
        expanded.append(node.path)
        completed = []
        for move in moves:
            first_child_visit = push(move)
            child = analyse_node_fn(tracker)
            child_entry = visit(child, depth + 1, first_child_visit)
            child_entry["move"] = move
            if depth == 0:
                operations.append({"op": "reset", "path": []})
                tracker.reset()
            else:
                operations.append({"op": "back", "path": list(tracker.path[:-1])})
                tracker.back()
            previous = [entry["move"] for entry in completed]
            if move in previous:
                completed[previous.index(move)] = child_entry
            else:
                completed.append(child_entry)
            operations.append(update_operation(node, completed))
        aggregate, _ = _aggregate(node.turn, root_turn, completed)
        return aggregate

    def adaptive_root():
        operations.append(evaluation_operation(root_node, 1, "initial"))
        operations.append(evaluation_operation(root_node, 8, "leaf"))
        if mate_in_one() is not None:
            visit(root_node, 0, first_visit=False)
            return
        if not _should_expand(spec, root_node, 0, len(visited)):
            return
        moves = _child_moves(spec, root_node)
        if not moves:
            raise TeacherError("An expanding root has no candidates")
        expanded.append(())
        nodes = {}
        values = {}
        proposed = []

        def write_root_update():
            completed = []
            for candidate in proposed:
                entry = dict(values[candidate])
                entry["move"] = candidate
                completed.append(entry)
            operations.append(update_operation(root_node, completed))

        for move in moves:
            push(move)
            child = analyse_node_fn(tracker)
            nodes[move] = child
            if child.terminal:
                operations.append(terminal_operation(child))
                value = terminal_entry(child)
            else:
                operations.append(evaluation_operation(child, 1, "initial"))
                operations.append(evaluation_operation(child, 8, "leaf"))
                value = value_entry(child, 8)
            operations.append({"op": "reset", "path": []})
            tracker.reset()
            proposed.append(move)
            value["move"] = move
            values[move] = value
            write_root_update()

        for target_depth in (2, 3):
            deep = [
                move
                for move in proposed
                if not nodes[move].terminal and _should_expand(spec, nodes[move], 1, 0)
            ]
            deep.sort(key=lambda move: -_instability(nodes[move]))
            reserve = (1 << target_depth) - 2
            selected = []
            budget = len(visited)
            for move in deep:
                if budget + reserve > spec.max_nodes:
                    continue
                selected.append(move)
                budget += reserve
            if not selected:
                break
            for move in selected:
                operations.append(
                    {"op": "move", "uci": move, "path": [move], "revisit": True}
                )
                tracker.push(move)
                value = visit(nodes[move], 1, first_visit=False)
                operations.append({"op": "reset", "path": []})
                tracker.reset()
                value["move"] = move
                values[move] = value
                write_root_update()

    if spec.kind == "adaptive":
        adaptive_root()
    else:
        visit(root_node, 0)
    if tracker.path != ():
        raise TeacherError("The teacher walk did not return to the root")
    states = derive_states(operations)
    if not states:
        raise TeacherError("The operation log has no non-final state")
    state = states[-1]
    if state.best is None:
        raise TeacherError("The root state has no decision")
    operations.append(
        {
            "op": "final",
            "uci": state.best,
            "path": [],
            "basis": state.basis,
            "value": state.value,
            "mate": state.mate,
            "mate_win": state.mate_win,
            "completed": [dict(entry) for entry in state.completed],
            "shallow": state.shallow,
            "candidates": _candidate_summary(state),
        }
    )
    return TreeResult(
        spec=spec,
        operations=tuple(operations),
        nodes=tuple(node_order),
        expanded=tuple(expanded),
        final_move=state.best,
        final_basis=state.basis,
        final_value=state.value,
    )


def plan_boards(root_fen: str, repetition: int, operations) -> tuple[int, ...]:
    """Indices of operations before which the tool copy requests BOARD.

    Only new, non-root expansions can request an observation. Re-entering an
    existing path is location rather than expansion: it never requests BOARD
    and restarts the departure clock. A non-root node is stale when it was left
    at least ``BOARD_RETURN_PLIES`` moves ago, or when ``BOARD_PLY_INTERVAL``
    moves have passed since the last observation. Staleness is consumed by the
    next new expansion; evaluations in between do not clear it. The root board
    prefix is always observed and the root never requests BOARD, so no request
    repeats without a new move.
    """
    tracker = PathTracker(root_fen, repetition)
    last_departure = {}
    pending_return = set()
    plies_since_observation = 0
    move_count = 0
    slots = []

    def mark_return(path):
        if not path:
            return
        departure = last_departure.get(path)
        if departure is not None and move_count - departure >= BOARD_RETURN_PLIES:
            pending_return.add(path)

    for index, operation in enumerate(operations):
        kind = operation["op"]
        if kind == "move":
            node = tracker.path
            revisit = bool(operation.get("revisit"))
            if (
                not revisit
                and node != ()
                and (
                    node in pending_return
                    or plies_since_observation >= BOARD_PLY_INTERVAL
                )
            ):
                slots.append(index)
                pending_return.discard(node)
                plies_since_observation = 0
            tracker.push(operation["uci"])
            move_count += 1
            plies_since_observation += 1
            last_departure[node] = move_count
            if not revisit:
                mark_return(tracker.path)
        elif kind == "back":
            last_departure[tracker.path] = move_count
            tracker.back()
            mark_return(tracker.path)
        elif kind == "reset":
            last_departure[tracker.path] = move_count
            tracker.reset()
            mark_return(tracker.path)
        elif kind == "evaluate":
            pass
        elif kind == "final":
            break
        else:
            raise TeacherError(f"Unknown operation: {kind}")
        if tuple(operation["path"]) != tracker.path:
            raise TeacherError(
                f"Operation path {operation['path']} disagrees with {tracker.path}"
            )
    return tuple(slots)


@dataclass(frozen=True)
class Emitted:
    """One emitted token stream and its mapping back to the operation log."""

    tokens: tuple[int, ...]
    roles: tuple[int, ...]
    steps: tuple
    op_spans: tuple[tuple[int, int], ...]
    board_spans: tuple[tuple[int, int, int, int], ...]
    step_index: tuple[int, ...]
    writer: TraceWriter


def emit_operations(
    root_fen: str,
    repetition: int,
    operations,
    board_slots,
    tool: bool,
    require_final: bool = True,
) -> Emitted:
    """Emit the operation log; the tool copy adds the planned BOARD requests."""
    slots = tuple(board_slots)
    if list(slots) != sorted(set(slots)):
        raise TeacherError("BOARD slots must be sorted and unique")
    writer = TraceWriter(root_fen, repetition)
    op_spans = []
    board_spans = []
    step_index = []
    for index, operation in enumerate(operations):
        if index in slots:
            if operation["op"] != "move":
                raise TeacherError("BOARD may only precede an expansion")
            if tool:
                start = len(writer.tokens)
                writer.board()
                board_spans.append(
                    (index, start, len(writer.tokens), len(writer.steps) - 1)
                )
        start = len(writer.tokens)
        kind = operation["op"]
        if kind == "move":
            expected = writer.path + (operation["uci"],)
            writer.move(operation["uci"])
            if writer.path != expected:
                raise TeacherError("Move did not advance to its declared path")
        elif kind == "back":
            writer.back()
        elif kind == "reset":
            writer.reset()
        elif kind == "evaluate":
            if operation.get("mate") is not None:
                writer.evaluate(mate=operation["mate"], win=operation["mate_win"])
            else:
                writer.evaluate(value=operation["value"])
        elif kind == "final":
            writer.final(operation["uci"])
        else:
            raise TeacherError(f"Unknown operation: {kind}")
        if tuple(operation["path"]) != writer.path:
            raise TeacherError(
                f"Operation path {operation['path']} disagrees with {writer.path}"
            )
        op_spans.append((start, len(writer.tokens)))
        step_index.append(len(writer.steps) - 1)
        if writer.finished and index != len(operations) - 1:
            raise TeacherError("FINAL must be the last operation")
    if require_final and not writer.finished:
        raise TeacherError("The operation log does not end with FINAL")
    if not require_final and writer.finished:
        raise TeacherError("A prefix cannot contain FINAL")
    return Emitted(
        tokens=tuple(int(token) for token in writer.tokens),
        roles=tuple(int(role) for role in writer.roles),
        steps=tuple(writer.steps),
        op_spans=tuple(op_spans),
        board_spans=tuple(board_spans),
        step_index=tuple(step_index),
        writer=writer,
    )


def build_states(operations, no_tool: Emitted, tool: Emitted) -> list[dict]:
    """Operation-boundary truncation states, including BOARD response ends."""
    root_states = derive_states(operations)
    states = []
    for index, state in enumerate(root_states):
        if state.best is None:
            raise TeacherError("A non-final operation boundary has no decision")
        states.append(
            {
                "op": index,
                "board": False,
                "tool": tool.op_spans[index][1],
                "no_tool": no_tool.op_spans[index][1],
                "path": list(operations[index]["path"]),
                "best": state.best,
                "basis": state.basis,
                "value": state.value,
                "mate": state.mate,
                "mate_win": state.mate_win,
            }
        )
    for op_index, start, end, step_index in tool.board_spans:
        if op_index < 1:
            raise TeacherError("A BOARD cannot precede the first operation")
        state = root_states[op_index - 1]
        if state.best is None:
            raise TeacherError("A BOARD response boundary has no decision")
        path = list(tool.steps[step_index].path)
        states.append(
            {
                "op": op_index,
                "board": True,
                "tool": end,
                "no_tool": no_tool.op_spans[op_index - 1][1],
                "path": path,
                "best": state.best,
                "basis": state.basis,
                "value": state.value,
                "mate": state.mate,
                "mate_win": state.mate_win,
            }
        )
    states.sort(key=lambda entry: (entry["tool"], entry["no_tool"]))
    previous_tool = -1
    previous_no_tool = -1
    for state in states:
        if state["tool"] < previous_tool:
            raise TeacherError("Truncation states are out of tool token order")
        if state["no_tool"] < previous_no_tool:
            raise TeacherError("Truncation states are out of no-tool token order")
        previous_tool = state["tool"]
        previous_no_tool = state["no_tool"]
    return states


def state_for_prefix(states, position: int, tool: bool = True):
    """The first operation boundary at or after a token prefix length.

    A position inside a BOARD response therefore lands at the response end,
    and a position inside any other operation lands after that operation.
    """
    if position <= 0 or not states:
        return None
    key = "tool" if tool else "no_tool"
    for state in states:
        if state[key] >= position:
            return state
    return states[-1]


def materialize_state(root_fen: str, repetition: int, operations, state, tool: bool):
    """Emit a truncation prefix and its state-derived FINAL move.

    The writer is returned so callers can inspect or replay its tokens. A
    BOARD response is either wholly present or wholly absent.
    """
    op_index = state["op"]
    if state["board"]:
        prefix = operations[:op_index]
        emitted = emit_operations(
            root_fen,
            repetition,
            prefix,
            plan_boards(root_fen, repetition, prefix),
            tool,
            require_final=False,
        )
        writer = emitted.writer
        key = "tool" if tool else "no_tool"
        if tool:
            if op_index not in plan_boards(
                root_fen, repetition, operations[: op_index + 1]
            ):
                raise TeacherError("The stored BOARD state is not in the board plan")
            writer.board()
        if len(writer.tokens) != state[key]:
            raise TeacherError("A truncation prefix length disagrees with its state")
    else:
        prefix = operations[: op_index + 1]
        emitted = emit_operations(
            root_fen,
            repetition,
            prefix,
            plan_boards(root_fen, repetition, prefix),
            tool,
            require_final=False,
        )
        writer = emitted.writer
        key = "tool" if tool else "no_tool"
        if len(writer.tokens) != state[key]:
            raise TeacherError("A truncation prefix length disagrees with its state")
    if state["best"] is None:
        raise TeacherError("A truncation state has no move")
    writer.final(state["best"])
    return writer


def zero_thought_tokens(root_fen: str, repetition: int, move: str) -> tuple[int, ...]:
    """The independent direct-answer sample: FINAL plus the full teacher move."""
    writer = TraceWriter(root_fen, repetition)
    writer.final(move)
    return tuple(int(token) for token in writer.tokens)


def _check_evaluation_token(program, step, operation):
    start = step.start
    if operation.get("mate") is not None:
        expected = mate_token(operation["mate"], operation["mate_win"])
    else:
        expected = VALUE + value_bin(operation["value"])
    if program.tokens[start] != expected:
        raise TeacherError(
            f"Evaluation token {program.tokens[start]} is not {expected} at {start}"
        )


def _compare_step(program, step, operation):
    kind = operation["op"]
    expected_operation = {
        "move": "move",
        "back": "back",
        "reset": "reset",
        "evaluate": "evaluate",
        "final": "final",
    }[kind]
    if step.operation != expected_operation:
        raise TeacherError(
            f"Step {step.start} is {step.operation}, expected {expected_operation}"
        )
    if tuple(operation["path"]) != step.path:
        raise TeacherError(f"Step path {step.path} disagrees with the operation")
    if kind == "move" and step.uci != operation["uci"]:
        raise TeacherError("Move step disagrees with the operation")
    if kind == "evaluate":
        if operation.get("mate") is not None:
            if (step.mate, step.mate_win) != (
                operation["mate"],
                operation["mate_win"],
            ):
                raise TeacherError("Mate step disagrees with the operation")
        elif step.mate is not None:
            raise TeacherError("A value operation decoded as a mate")
        _check_evaluation_token(program, step, operation)
    if kind == "final" and step.uci != operation["uci"]:
        raise TeacherError("FINAL step disagrees with the operation")


def _check_node_operation(node: NodeAnalysis, operation):
    source = operation["source"]
    if source == "initial":
        depth = 1
        if node.terminal:
            raise TeacherError("A terminal node cannot have an initial evaluation")
    elif source == "leaf":
        depth = 8
        if node.terminal:
            raise TeacherError("A terminal node cannot have a leaf evaluation")
    elif source == "terminal":
        if not node.terminal:
            raise TeacherError("A non-terminal node cannot have a terminal rule")
        if operation["value"] != node.rule_value:
            raise TeacherError("Terminal value disagrees with its node")
        if (
            operation["mate"],
            operation["mate_win"],
        ) != (node.rule_mate, node.rule_mate_win):
            raise TeacherError("Terminal mate disagrees with its node")
        return
    else:
        raise TeacherError(f"Unknown evaluation source: {source}")
    if operation["d"] != depth:
        raise TeacherError("Evaluation depth disagrees with its source")
    value = root_value(node, depth)
    mate, mate_win = root_mate(node, depth)
    visible = _visible_entry(None, value, mate, mate_win)
    if operation["value"] != visible["value"]:
        raise TeacherError("Evaluation value disagrees with its visible token")
    if operation.get("raw_value") != value:
        raise TeacherError("Evaluation raw value disagrees with its node")
    if (operation["mate"], operation["mate_win"]) != (mate, mate_win):
        raise TeacherError("Evaluation mate disagrees with its node")
    if operation["rank1"] != node.rank1(depth).move:
        raise TeacherError("Evaluation rank 1 disagrees with its node")


def verify_teacher_trace(
    root_fen: str,
    repetition: int,
    operations,
    node_records,
    no_tool: Emitted,
    tool: Emitted,
    states,
    spec,
    materialized_states: int = 3,
) -> dict:
    """Replay and numerically verify one teacher trace.

    Tokens are replayed with python-chess; every operation is matched to its
    replay step, parent updates are reconstructed from prior child values, the
    FINAL decision is recomputed from the root state, and representative
    truncation prefixes are materialized and replayed.
    """
    no_program = replay(no_tool.tokens, root_fen, repetition)
    tool_program = replay(tool.tokens, root_fen, repetition)
    if len(no_program.steps) != len(operations):
        raise TeacherError("No-tool replay steps do not match the operation log")
    if len(tool_program.steps) != len(operations) + len(tool.board_spans):
        raise TeacherError("Tool replay steps do not match the operation log")
    for index, operation in enumerate(operations):
        no_step = no_program.steps[index]
        tool_step = tool_program.steps[tool.step_index[index]]
        if no_tool.op_spans[index] != (no_step.start, no_step.end):
            raise TeacherError("No-tool operation span disagrees with replay")
        if tool.op_spans[index] != (tool_step.start, tool_step.end):
            raise TeacherError("Tool operation span disagrees with replay")
        _compare_step(no_program, no_step, operation)
        _compare_step(tool_program, tool_step, operation)

    for op_index, start, end, step_index in tool.board_spans:
        step = tool_program.steps[step_index]
        if (step.start, step.end) != (start, end) or step.operation != "board":
            raise TeacherError("BOARD step does not match its recorded span")
        operation = operations[op_index]
        if operation["op"] != "move":
            raise TeacherError("BOARD is only allowed before an expansion")
        if step.path != tuple(operation["path"][:-1]):
            raise TeacherError("BOARD observation is at the wrong node")

    stripped_tokens = list(tool.tokens)
    stripped_roles = list(tool.roles)
    for op_index, start, end, step_index in reversed(tool.board_spans):
        del stripped_tokens[start:end]
        del stripped_roles[start:end]
    if tuple(stripped_tokens) != no_tool.tokens:
        raise TeacherError("Tool and no-tool traces differ outside BOARD spans")
    if tuple(stripped_roles) != no_tool.roles:
        raise TeacherError("Tool and no-tool roles differ outside BOARD spans")

    nodes = {
        tuple(record["path"]): NodeAnalysis.from_record(record)
        for record in node_records
    }
    for operation in operations:
        if operation["op"] != "evaluate":
            continue
        path = tuple(operation["path"])
        node = nodes.get(path)
        if node is None:
            raise TeacherError(f"Evaluation at an unrecorded node: {path}")
        if operation["source"] != "update":
            _check_node_operation(node, operation)

    current = ()
    last_eval = {}
    completed = {}
    children = {}
    root_completed = []
    proposed = []
    shallow = None
    shallow_value = None
    shallow_mate = None
    shallow_win = None
    pending = None
    expect_entry = None
    expect_update = None
    visited = {()}
    final_operation = None
    root_turn = chess.Board(root_fen).turn

    def evaluation_entry(operation, move=None):
        return _entry(
            move,
            operation["value"],
            operation["mate"],
            operation["mate_win"],
            operation.get("raw_value"),
        )

    for index, operation in enumerate(operations):
        kind = operation["op"]
        if expect_entry is not None and kind != "evaluate":
            raise TeacherError("An entered node did not finish d1/d8 evaluations")
        if expect_update is not None and (
            kind != "evaluate"
            or operation.get("source") != "update"
            or tuple(operation["path"]) != expect_update
        ):
            raise TeacherError("A returned child did not update its parent")
        if kind == "move":
            path = tuple(operation["path"])
            if path != current + (operation["uci"],):
                raise TeacherError("A move does not extend the current path")
            revisit = bool(operation.get("revisit"))
            if path in visited:
                if not revisit:
                    raise TeacherError("A node was re-entered without a revisit mark")
                completed[path] = []
            else:
                if revisit:
                    raise TeacherError("A revisit marked an unvisited node")
                visited.add(path)
                expect_entry = (
                    ["terminal"] if nodes[path].terminal else ["initial", "leaf"]
                )
                if len(path) == 1:
                    if operation["uci"] in proposed:
                        raise TeacherError("A root candidate was proposed twice")
                    proposed.append(operation["uci"])
                children.setdefault(current, []).append(path[-1])
            current = path
        elif kind in {"back", "reset"}:
            if pending is not None:
                raise TeacherError("A completed child was not updated")
            child = current
            path = tuple(operation["path"])
            if kind == "reset":
                if len(child) != 1 or path != ():
                    raise TeacherError("RESET must return to the root from depth 1")
            elif not child or path != child[:-1]:
                raise TeacherError("BACK must return to the parent")
            pending = (path, child)
            expect_update = path
            current = path
        elif kind == "evaluate":
            path = tuple(operation["path"])
            if path != current:
                raise TeacherError("An evaluation is not at the current node")
            source = operation["source"]
            if expect_entry is not None:
                if path != current or source != expect_entry.pop(0):
                    raise TeacherError(
                        "A node did not write d1/d8 in the required order"
                    )
                if not expect_entry:
                    expect_entry = None
            if source in {"initial", "leaf", "terminal"}:
                if source == "terminal" and path == ():
                    raise TeacherError("A root position cannot be terminal")
                last_eval[path] = evaluation_entry(operation)
                if path == () and source in {"initial", "leaf"}:
                    shallow = operation["rank1"]
                    shallow_value = operation["value"]
                    shallow_mate = operation["mate"]
                    shallow_win = operation["mate_win"]
            elif source == "update":
                if pending is None:
                    raise TeacherError("An update has no completed child")
                parent, child = pending
                if path != parent or child not in last_eval:
                    raise TeacherError("An update is not at the returned parent")
                if parent == ():
                    entries = operation["completed"]
                    if not entries:
                        raise TeacherError("A root update needs a candidate")
                    moves = [entry["move"] for entry in entries]
                    if moves != proposed[: len(moves)]:
                        raise TeacherError("A root update is not a candidate prefix")
                    expected = []
                    for move in moves:
                        key = (move,)
                        if key not in last_eval:
                            raise TeacherError("A root candidate has no evaluation")
                        entry = dict(last_eval[key])
                        entry["move"] = move
                        expected.append(entry)
                    root_completed = expected
                else:
                    entry = dict(last_eval[child])
                    entry["move"] = child[-1]
                    expected = list(completed.get(parent, []))
                    previous = [item["move"] for item in expected]
                    if child[-1] in previous:
                        expected[previous.index(child[-1])] = entry
                    else:
                        expected.append(entry)
                if operation["completed"] != expected:
                    raise TeacherError(
                        f"An update does not include exactly prior children at {parent}: "
                        f"stored={operation['completed']!r} expected={expected!r}"
                    )
                aggregate, chosen = _aggregate(nodes[parent].turn, root_turn, expected)
                for field in ("value", "mate", "mate_win"):
                    if operation[field] != aggregate[field]:
                        raise TeacherError(
                            "An update value or mate is not the aggregate"
                        )
                if operation["chosen"] != chosen:
                    raise TeacherError("An update chose the wrong child")
                completed[parent] = expected
                aggregate["move"] = parent[-1] if parent else None
                last_eval[parent] = aggregate
                pending = None
                expect_update = None
            else:
                raise TeacherError(f"Unknown evaluation source: {source}")
        elif kind == "final":
            if pending is not None:
                raise TeacherError("FINAL follows an unupdated child")
            final_operation = operation
            break
        else:
            raise TeacherError(f"Unknown operation: {kind}")
    if final_operation is None:
        raise TeacherError("The operation log has no FINAL")
    if pending is not None or expect_entry is not None:
        raise TeacherError("The operation log ends with unfinished node work")

    for parent, entries in completed.items():
        if not entries:
            continue
        actual = [entry["move"] for entry in entries]
        intended = list(_child_moves(spec, nodes[parent]))
        if actual == intended:
            continue
        board = chess.Board(root_fen)
        for move in parent:
            board.push_uci(move)
        mate = None
        for move in board.legal_moves:
            candidate = board.copy()
            candidate.push(move)
            if candidate.is_checkmate():
                mate = move.uci()
                break
        if actual != [mate]:
            raise TeacherError(f"A parent update is not a complete subtree: {parent}")

    if proposed:
        if not root_completed:
            raise TeacherError("An expanded root has no completed candidates")
        if [entry["move"] for entry in root_completed] != proposed:
            raise TeacherError("An expanded root left a candidate incomplete")
        chosen = max(
            root_completed, key=lambda entry: _prefer(entry, root_completed[0])
        )
        for field in ("uci", "value", "mate", "mate_win"):
            expected = chosen["move"] if field == "uci" else chosen[field]
            if final_operation[field] != expected:
                raise TeacherError("FINAL disagrees with the visible root preference")
        if final_operation["basis"] != "completed":
            raise TeacherError(
                "FINAL from an expanded root must use completed candidates"
            )
        if final_operation["completed"] != root_completed:
            raise TeacherError("FINAL does not carry the completed root candidates")
    else:
        if root_completed:
            raise TeacherError("An unexpanded root cannot have completed candidates")
        if final_operation["basis"] != "shallow":
            raise TeacherError("FINAL from an unexpanded root must be shallow")
        if shallow is None or final_operation["uci"] != shallow:
            raise TeacherError("FINAL is not the root d8 rank 1")
        if final_operation["value"] != shallow_value:
            raise TeacherError("FINAL value disagrees with the shallow root")
        if (final_operation["mate"], final_operation["mate_win"]) != (
            shallow_mate,
            shallow_win,
        ):
            raise TeacherError("FINAL mate disagrees with the shallow root")
    expected_candidates = _candidate_summary(
        RootState(
            proposed=tuple(proposed),
            completed=tuple(root_completed),
            shallow=shallow,
            shallow_value=shallow_value,
            best=final_operation["uci"],
            basis=final_operation["basis"],
            value=final_operation["value"],
            mate=final_operation["mate"],
            mate_win=final_operation["mate_win"],
        )
    )
    if final_operation["candidates"] != expected_candidates:
        raise TeacherError("FINAL candidate completion summary is inconsistent")

    expected_states = build_states(operations, no_tool, tool)
    if states != expected_states:
        raise TeacherError("Stored truncation states do not match the operation log")
    root_board = chess.Board(root_fen)
    legal = {move.uci() for move in root_board.legal_moves}
    for state in states:
        if state["best"] not in legal:
            raise TeacherError("A truncation state has an illegal root move")
        if not 0 < state["tool"] <= len(tool.tokens):
            raise TeacherError("A tool truncation index is outside its trace")
        if not 0 < state["no_tool"] <= len(no_tool.tokens):
            raise TeacherError("A no-tool truncation index is outside its trace")
    for op_index, start, end, step_index in tool.board_spans:
        for state in states:
            if start < state["tool"] < end:
                raise TeacherError("A truncation state cuts a BOARD response")

    materialized = 0
    if states and materialized_states:
        low = max(1, int(len(tool.tokens) * 0.05))
        high = max(low, int(len(tool.tokens) * 0.95))
        candidates = [state for state in states if low <= state["tool"] <= high]
        if candidates:
            picks = {0, len(candidates) // 2, len(candidates) - 1}
            for pick in sorted(picks)[:materialized_states]:
                state = candidates[pick]
                writer = materialize_state(
                    root_fen, repetition, operations, state, tool=True
                )
                program = replay(writer.tokens, root_fen, repetition)
                if program.final_move != state["best"]:
                    raise TeacherError("A materialized truncation changed its answer")
                materialized += 1
    return {
        "operations": len(operations),
        "tool_steps": len(tool_program.steps),
        "board_steps": len(tool.board_spans),
        "states": len(states),
        "materialized_states": materialized,
    }


def synthesize_root(
    engine,
    root: Mapping,
    config: AnnotationConfig,
    specs=DEFAULT_TEACHER_SPECS,
) -> dict:
    """Synthesize all requested teachers for one selected root.

    One analysis cache serves every teacher, and every node count is by path.
    A teacher tree is built, emitted, and verified before the next teacher
    starts; the returned record contains raw node annotations (node side to
    move) and operation logs (root player perspective).
    """
    root_fen = root["fen"]
    repetition = root.get("repetition", 1)
    board = chess.Board(root_fen)
    if board.is_game_over():
        raise TeacherError(f"Selected root is terminal: {root['root_id']}")
    start = time.monotonic()
    cache = AnalysisCache(engine, config, board.turn)
    teachers = {}
    total_states = 0
    for spec in specs:
        teacher_start = time.monotonic()
        tree = simulate_tree(spec, root_fen, repetition, cache)
        board_slots = plan_boards(root_fen, repetition, tree.operations)
        no_tool = emit_operations(
            root_fen, repetition, tree.operations, board_slots, tool=False
        )
        tool = emit_operations(
            root_fen, repetition, tree.operations, board_slots, tool=True
        )
        states = build_states(tree.operations, no_tool, tool)
        verification = verify_teacher_trace(
            root_fen,
            repetition,
            tree.operations,
            cache.records(),
            no_tool,
            tool,
            states,
            spec,
        )
        total_states += len(states)
        analysed_paths = {
            tuple(operation["path"])
            for operation in tree.operations
            if operation["op"] == "evaluate" and operation["source"] == "initial"
        }
        node_by_path = cache.nodes_by_path()
        engine_calls = len(analysed_paths)
        analysis_ms = sum(node_by_path[path].elapsed_ms for path in analysed_paths)
        response_tokens = sum(
            end - start_index - 1 for _, start_index, end, _ in tool.board_spans
        )
        model_tokens = len(tool.tokens) - response_tokens
        metrics = _teacher_metrics(
            tree,
            no_tool,
            tool,
            states,
            board_slots,
            response_tokens,
            model_tokens,
            engine_calls,
            analysis_ms,
            int((time.monotonic() - teacher_start) * 1000),
        )
        teachers[spec.name] = {
            "name": spec.name,
            "spec": spec.as_dict(),
            "operations": list(tree.operations),
            "board_slots": list(board_slots),
            "final_move": tree.final_move,
            "final_basis": tree.final_basis,
            "final_value": tree.final_value,
            "no_tool": {
                "tokens": list(no_tool.tokens),
                "length": len(no_tool.tokens),
            },
            "tool": {
                "tokens": list(tool.tokens),
                "length": len(tool.tokens),
            },
            "truncation": {
                "states": states,
                "final_start": {
                    "no_tool": no_tool.op_spans[-1][0],
                    "tool": tool.op_spans[-1][0],
                },
                "board_spans": [
                    [start_index, end] for _, start_index, end, _ in tool.board_spans
                ],
            },
            "metrics": metrics,
            "verification": verification,
        }
    record = {
        "root_id": root["root_id"],
        "split": root["split"],
        "fen": root_fen,
        "repetition": repetition,
        "turn": "white" if board.turn else "black",
        "source": root.get("source"),
        "d16": root.get("d16"),
        "nodes": cache.records(),
        "teachers": teachers,
        "analysis": {
            "calls": cache.engine_calls,
            "elapsed_ms": cache.analysis_ms,
            "nodes": len(cache),
            "terminal_nodes": sum(
                1 for node in cache.nodes_by_path().values() if node.terminal
            ),
        },
        "elapsed_ms": int((time.monotonic() - start) * 1000),
        "states": total_states,
    }
    return record


def _teacher_metrics(
    tree,
    no_tool,
    tool,
    states,
    board_slots,
    response_tokens,
    model_tokens,
    engine_calls,
    analysis_ms,
    build_ms,
):
    operations = tree.operations
    node_paths = [
        tuple(operation["path"])
        for operation in operations
        if operation["op"] == "evaluate"
        and operation["source"] in {"initial", "terminal"}
    ]
    terminal_paths = [
        tuple(operation["path"])
        for operation in operations
        if operation["op"] == "evaluate" and operation["source"] == "terminal"
    ]
    move_ops = [
        operation
        for operation in operations
        if operation["op"] == "move" and not operation.get("revisit")
    ]
    revisits = sum(
        1
        for operation in operations
        if operation["op"] == "move" and operation.get("revisit")
    )
    children = {}
    for operation in move_ops:
        parent = tuple(operation["path"][:-1])
        children[parent] = children.get(parent, 0) + 1
    expanded = set(children)
    leaf_paths = [path for path in node_paths if path not in expanded]
    root_moves = children.get((), 0)
    proposed = [
        operation["uci"] for operation in move_ops if len(operation["path"]) == 1
    ]
    updates = 0
    revisions = 0
    last_eval = {}
    leaf_shift = 0
    leaf_count = 0
    initial_values = {}
    for operation in operations:
        if operation["op"] != "evaluate":
            continue
        path = tuple(operation["path"])
        source = operation["source"]
        if source == "initial":
            initial_values[path] = operation["value"]
        elif source == "leaf":
            if path not in expanded:
                leaf_count += 1
                if initial_values.get(path) != operation["value"]:
                    leaf_shift += 1
        elif source == "update":
            updates += 1
            if last_eval.get(path) != operation["value"]:
                revisions += 1
        last_eval[path] = operation["value"]
    state_changes = 0
    for previous, current in zip(states, states[1:]):
        if previous["best"] != current["best"]:
            state_changes += 1
    stop_depths = {}
    for path in leaf_paths:
        stop_depths[str(len(path))] = stop_depths.get(str(len(path)), 0) + 1
    final_basis = tree.final_basis
    overturned = bool(proposed) and tree.final_move != proposed[0]
    return {
        "nodes": len(node_paths),
        "terminal_nodes": len(terminal_paths),
        "analysed_nodes": len(node_paths) - len(terminal_paths),
        "expanded_nodes": len(expanded),
        "non_root_expanded_nodes": sum(1 for path in expanded if path),
        "max_depth": max((len(path) for path in node_paths), default=0),
        "max_width": max(children.values(), default=0),
        "root_width": root_moves,
        "proposed": len(proposed),
        "revisits": revisits,
        "updates": updates,
        "revisions": revisions,
        "leaf_count": leaf_count,
        "leaf_shift": leaf_shift,
        "states": len(states),
        "state_transitions": max(0, len(states) - 1),
        "state_changes": state_changes,
        "final_basis": final_basis,
        "final_move": tree.final_move,
        "final_value": tree.final_value,
        "overturned": overturned,
        "stop_depths": stop_depths,
        "board_requests": len(board_slots),
        "board_response_tokens": response_tokens,
        "no_tool_tokens": len(no_tool.tokens),
        "tool_tokens": len(tool.tokens),
        "model_tokens": model_tokens,
        "token_increase": len(tool.tokens) - len(no_tool.tokens),
        "engine_calls": engine_calls,
        "analysis_ms": analysis_ms,
        "build_ms": build_ms,
    }


def restricted_value(engine, board: chess.Board, move: str, depth: int) -> dict:
    """A d``depth`` search restricted to one root move, in root-player terms."""
    parsed = chess.Move.from_uci(move)
    if parsed not in board.legal_moves:
        raise TeacherError(f"Restricted search move is illegal: {move}")
    best = None
    reported = 0
    with engine.analysis(
        board,
        chess.engine.Limit(depth=depth),
        multipv=1,
        game=object(),
        root_moves=[parsed],
        options={"UCI_ShowWDL": True},
    ) as stream:
        for info in stream:
            if "depth" not in info or "score" not in info or not info.get("pv"):
                continue
            if info.get("multipv", 1) != 1:
                continue
            if info.get("lowerbound") or info.get("upperbound"):
                continue
            reported = max(reported, info["depth"])
            if info["depth"] < depth:
                continue
            score = info["score"].pov(board.turn)
            wdl = info["wdl"].pov(board.turn)
            total = wdl.total()
            if total < 1:
                raise TeacherError("Engine returned an empty WDL score")
            best = {
                "move": move,
                "depth": info["depth"],
                "cp": None if score.is_mate() else score.score(),
                "mate": score.mate(),
                "value": (wdl.wins - wdl.losses) / total,
                "wdl": [wdl.wins, wdl.draws, wdl.losses],
                "pv": [item.uci() for item in info["pv"]],
            }
    if best is None or reported < depth:
        raise TeacherError(
            f"Restricted d{depth} search of {move} stopped at {reported}"
        )
    return best
