"""Reusable Stockfish scores for legal moves, kept outside model inference."""

import hashlib
import json
import shutil
import sqlite3
from contextlib import ExitStack
from pathlib import Path

import chess
import chess.engine

from pondera.chess.codec import cap_mate_distance, value_bin


def score_key(score):
    mate = score["mate"]
    return (0, score["cp"]) if mate is None else (1 if mate > 0 else -1, -mate)


def can_claim(board, repetition):
    return repetition >= 3 or board.halfmove_clock >= 100


class Reference:
    def __init__(self, executable, cache, *, nodes, threads=1, hash_mb=64):
        executable = Path(shutil.which(str(executable)) or executable).resolve()
        with executable.open("rb") as handle:
            binary = hashlib.file_digest(handle, "sha256").hexdigest()
        with ExitStack() as resources:
            self.engine = chess.engine.SimpleEngine.popen_uci(
                str(executable), timeout=30
            )
            resources.callback(self.engine.quit)
            self.engine.configure(
                {"Threads": threads, "Hash": hash_mb, "UCI_ShowWDL": True}
            )
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
            self.engine.quit()
        finally:
            self.cache.close()

    def scores(self, board):
        """Evaluate each legal root move with fresh search state and equal nodes.

        History is part of the key. The root's prior repetition count is not
        fabricated as move history: this matches the corpus teacher's FEN input.
        """
        position = json.dumps(
            [
                board.root().fen(en_passant="fen"),
                [move.uci() for move in board.move_stack],
            ]
        )
        cached = self.cache.execute(
            "SELECT scores FROM reference WHERE engine=? AND position=?",
            (self.identity, position),
        ).fetchone()
        if cached:
            self.hits += 1
            return json.loads(cached[0])
        result = {}
        for move in sorted(board.legal_moves, key=lambda move: move.uci()):
            info = self.engine.analyse(
                board,
                chess.engine.Limit(nodes=self.nodes),
                root_moves=[move],
                game=object(),
                info=chess.engine.INFO_ALL,
            )
            score = info["score"].pov(board.turn)
            mate = score.mate()
            if mate is not None:
                value = 1.0 if score > chess.engine.Cp(0) else -1.0
            else:
                wdl = info["wdl"].pov(board.turn)
                value = (wdl.wins - wdl.losses) / 1000
            result[move.uci()] = {"cp": score.score(), "mate": mate, "value": value}
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

    def grade(self, fen, repetition, answer, *, pv_plies=0):
        """Root move quality and root-value accuracy for a valid answer."""
        if not answer.valid:
            raise ValueError("Only valid answers receive engine quality scores")
        board = chess.Board(fen)
        root_turn = board.turn
        scores = self.scores(board)
        searched_best = max(scores.values(), key=score_key)
        root_value = searched_best["value"]
        if can_claim(board, repetition):
            scores = {**scores, "<claim_draw>": {"cp": 0, "mate": None, "value": 0.0}}
        best = max(scores.values(), key=score_key)
        chosen = scores[answer.decision]
        rank = 1 + sum(
            score_key(score) > score_key(chosen) for score in scores.values()
        )
        result = {
            "reference": best,
            "chosen": chosen,
            "move_rank": rank,
            "move_rank_fraction": (rank - 1) / max(len(scores) - 1, 1),
            "best_move": rank == 1,
            "cp_loss": best["cp"] - chosen["cp"]
            if best["cp"] is not None and chosen["cp"] is not None
            else None,
            "expected_score_loss": (best["value"] - chosen["value"]) / 2,
            "root_value": root_value,
            "value_absolute_error": abs(answer.value - root_value),
            "value_bin_correct": answer.value_bin == value_bin(root_value),
            "mate_distance_correct": (
                answer.mate_distance == cap_mate_distance(searched_best["mate"])
                if searched_best["mate"] is not None
                else None
            ),
            "pv": [],
        }
        if answer.decision != "<claim_draw>":
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
