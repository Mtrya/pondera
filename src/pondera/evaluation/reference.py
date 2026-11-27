"""Reusable Stockfish scores for legal moves, kept outside model inference."""

import hashlib
import json
import shutil
import sqlite3
import time
from contextlib import ExitStack
from pathlib import Path

from pondera.chess import _rules as rules
from pondera.chess.codec import cap_mate_distance, value_bin
from pondera.uci_process import UciError, UciProcess


def score_key(score):
    mate = score["mate"]
    return (0, score["cp"]) if mate is None else (1 if mate > 0 else -1, -mate)


def can_claim(board, repetition):
    return repetition >= 3 or board.halfmove_clock >= 100


class _ReferenceEngine(UciProcess):
    def __init__(self, path, threads, hash_mb, timeout=30):
        super().__init__(path, timeout)
        try:
            deadline = time.monotonic() + timeout
            self._send("uci")
            _, banner = self._read_until("uciok", deadline)
            self.id = {}
            options = set()
            for line in banner:
                if line.startswith("id "):
                    _, key, value = line.split(maxsplit=2)
                    self.id[key] = value
                elif line.startswith("option name "):
                    name, _, _ = line.removeprefix("option name ").partition(" type ")
                    options.add(name)
            for name, value in (
                ("Threads", threads),
                ("Hash", hash_mb),
                ("UCI_ShowWDL", "true"),
            ):
                if name not in options:
                    raise UciError(f"Engine does not expose {name}")
                self._send(f"setoption name {name} value {value}")
        except BaseException:
            self.close(force=True)
            raise

    def search(self, root_fen, history, nodes, move):
        deadline = time.monotonic() + self.timeout
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok", deadline)
        position = f"position fen {root_fen}"
        if history:
            position += " moves " + " ".join(history)
        self._send(position)
        self._send(f"go nodes {nodes} searchmoves {move}")
        bestmove, lines = self._read_until("bestmove", deadline)
        if bestmove.split()[1:2] != [move]:
            raise UciError(f"Expected bestmove {move}, received {bestmove!r}")
        score = None
        wdl = None
        try:
            for line in lines:
                words = line.split()
                if words[:1] != ["info"] or words[1:2] == ["string"]:
                    continue
                # PV contents are moves through the end of the line.
                if "pv" in words:
                    words = words[: words.index("pv")]
                if "score" in words:
                    index = words.index("score")
                    kind, value = words[index + 1 : index + 3]
                    if kind in {"cp", "mate"}:
                        score = kind, int(value)
                if "wdl" in words:
                    index = words.index("wdl")
                    wins, _, losses = map(int, words[index + 1 : index + 4])
                    wdl = (wins - losses) / 1000
        except ValueError as error:
            raise UciError("Malformed UCI score or WDL") from error
        if score is None:
            raise UciError("Engine returned no score")
        kind, value = score
        if kind == "mate":
            return {"cp": None, "mate": value, "value": 1.0 if value > 0 else -1.0}
        if wdl is None:
            raise UciError("Engine returned no WDL")
        return {"cp": value, "mate": None, "value": wdl}


class Reference:
    def __init__(self, executable, cache, *, nodes, threads=1, hash_mb=64):
        executable = Path(shutil.which(str(executable)) or executable).resolve()
        with executable.open("rb") as handle:
            binary = hashlib.file_digest(handle, "sha256").hexdigest()
        with ExitStack() as resources:
            self.engine = _ReferenceEngine(executable, threads, hash_mb)
            resources.callback(self.engine.close)
            self.settings = {
                "engine": self.engine.id,
                "nodes_per_move": nodes,
                "threads": threads,
                "hash_mb": hash_mb,
            }
            # Internal cache identity also distinguishes builds sharing an engine name.
            self.identity = json.dumps(
                {**self.settings, "binary": binary}, sort_keys=True
            )
            self.nodes = nodes
            self.cache = sqlite3.connect(cache)
            resources.callback(self.cache.close)
            self.cache.execute(
                "CREATE TABLE IF NOT EXISTS reference (engine TEXT, position TEXT, scores TEXT, PRIMARY KEY (engine, position))"
            )
            resources.pop_all()
        self.hits = self.searches = 0

    def close(self):
        try:
            self.engine.close()
        finally:
            self.cache.close()

    def scores(self, board):
        """Evaluate each legal root move with fresh search state and equal nodes.

        History is part of the key. The root's prior repetition count is not
        fabricated as move history: this matches the corpus teacher's FEN input.
        """
        root_fen = board.root().fen(en_passant="fen")
        history = [move.uci() for move in board.move_stack]
        position = json.dumps([root_fen, history])
        cached = self.cache.execute(
            "SELECT scores FROM reference WHERE engine=? AND position=?",
            (self.identity, position),
        ).fetchone()
        if cached:
            self.hits += 1
            return json.loads(cached[0])
        result = {}
        for move in sorted(move.uci() for move in board.legal_moves):
            result[move] = self.engine.search(root_fen, history, self.nodes, move)
            self.searches += 1
        if not result:
            raise ValueError("Engine reference needs a root with legal moves")
        self.cache.execute(
            "INSERT OR IGNORE INTO reference VALUES (?, ?, ?)",
            (self.identity, position, json.dumps(result)),
        )
        self.cache.commit()
        return json.loads(
            self.cache.execute(
                "SELECT scores FROM reference WHERE engine=? AND position=?",
                (self.identity, position),
            ).fetchone()[0]
        )

    def grade_decision(self, fen, repetition, decision):
        """Grade a legal move or draw claim independently of any answer suffix."""
        board = rules.Board(fen)
        if (
            not board.is_valid()
            or board.is_game_over()
            or type(repetition) is not int
            or not 1 <= repetition < 5
        ):
            raise ValueError("Decision grading needs a valid playable root")
        if decision == "<claim_draw>":
            if not can_claim(board, repetition):
                raise ValueError("The position does not permit a draw claim")
        else:
            try:
                move = rules.Move.from_uci(decision)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Decision must be a legal UCI move or draw claim"
                ) from error
            if not board.is_legal(move):
                raise ValueError("Decision is not legal in this position")
        scores = self.scores(board)
        searched_best = max(scores.values(), key=score_key)
        if can_claim(board, repetition):
            scores = {**scores, "<claim_draw>": {"cp": 0, "mate": None, "value": 0.0}}
        best = max(scores.values(), key=score_key)
        chosen = scores[decision]
        rank = 1 + sum(
            score_key(score) > score_key(chosen) for score in scores.values()
        )
        return {
            "reference": best,
            "chosen": chosen,
            "move_rank": rank,
            "move_rank_fraction": (rank - 1) / max(len(scores) - 1, 1),
            "best_move": rank == 1,
            "cp_loss": best["cp"] - chosen["cp"]
            if best["cp"] is not None and chosen["cp"] is not None
            else None,
            "expected_score_loss": (best["value"] - chosen["value"]) / 2,
            "root_value": searched_best["value"],
            "root_mate": searched_best["mate"],
        }

    def grade(self, fen, repetition, answer, *, pv_plies=0):
        """Move, value and PV quality for a complete valid answer."""
        if not answer.valid:
            raise ValueError("Only valid answers receive full answer quality scores")
        result = self.grade_decision(fen, repetition, answer.decision)
        root_value = result["root_value"]
        result.update(
            {
                "value_absolute_error": abs(answer.value - root_value),
                "value_bin_correct": answer.value_bin == value_bin(root_value),
                "mate_distance_correct": (
                    answer.mate_distance == cap_mate_distance(result["root_mate"])
                    if result["root_mate"] is not None
                    else None
                ),
                "pv": [],
            }
        )
        if answer.decision != "<claim_draw>":
            board = rules.Board(fen)
            root_turn = board.turn
            board.push_uci(answer.decision)
            for move in answer.pv[:pv_plies]:
                options = self.scores(board)
                preferred = max(options.values(), key=score_key)
                selected = options[move]
                result["pv"].append(
                    {
                        "move": move,
                        "root_player": board.turn == root_turn,
                        "cp_loss": preferred["cp"] - selected["cp"]
                        if preferred["cp"] is not None and selected["cp"] is not None
                        else None,
                        "best_move": score_key(selected) == score_key(preferred),
                    }
                )
                board.push_uci(move)
        return result
