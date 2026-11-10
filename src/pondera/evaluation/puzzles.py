"""Graded puzzle evaluation on the zero-presentation generation entry.

A puzzle's stored FEN precedes the opponent setup move. The solver's
moves are the odd indices of the solution; the opponent replies are played
as given. First-move accuracy is the primary metric, whole-line accuracy
the secondary one. Buckets are declared at execution time, not frozen here.
"""

import chess


def solve_puzzle(choose_move, row):
    """Return (first_move_correct, whole_line_correct) for one puzzle row."""
    board = chess.Board(row["FEN"])
    moves = row["Moves"].split()
    board.push_uci(moves[0])
    first_correct = False
    for index in range(1, len(moves), 2):
        try:
            candidate = chess.Move.from_uci(choose_move(board.fen()))
        except ValueError:
            return first_correct, False
        if candidate not in board.legal_moves:
            return first_correct, False
        board.push(candidate)
        correct = candidate.uci() == moves[index] or board.is_checkmate()
        if index == 1:
            first_correct = correct
        if not correct:
            return first_correct, False
        if board.is_checkmate() or index + 2 >= len(moves):
            return True, True
        board.push_uci(moves[index + 1])
    return first_correct, first_correct


def evaluate_puzzles(
    choose_move, rows, buckets=((0, 1200), (1200, 1600), (1600, 2000), (2000, 10**9))
):
    """First-move and whole-line accuracy per rating bucket."""
    report = {}
    for low, high in buckets:
        report[f"{low}-{high}"] = {"first": 0, "whole": 0, "count": 0}
    overall = {"first": 0, "whole": 0, "count": 0}
    for row in rows:
        first, whole = solve_puzzle(choose_move, row)
        overall["first"] += first
        overall["whole"] += whole
        overall["count"] += 1
        for low, high in buckets:
            if low <= row["Rating"] < high:
                bucket = report[f"{low}-{high}"]
                bucket["first"] += first
                bucket["whole"] += whole
                bucket["count"] += 1
                break
    for stats in [*report.values(), overall]:
        count = stats["count"]
        stats["first_accuracy"] = stats["first"] / count if count else None
        stats["whole_accuracy"] = stats["whole"] / count if count else None
    return {"buckets": report, "overall": overall}
