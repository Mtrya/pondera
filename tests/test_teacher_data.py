import json
import shutil
import tempfile
import unittest
from hashlib import file_digest
from pathlib import Path

import chess
from teacher_release import play, write_teacher_release

from pondera.chess.board_tokens import decode_board
from pondera.chess.mapping import UCI_MOVE_TO_IDX
from pondera.data.annotation import AnnotationConfig, annotate_positions
from pondera.data.positions import TokenizedPositions
from pondera.data.storage import write_manifest
from pondera.data.teacher import prepare_teacher_policy


class TeacherReleaseShapeTests(unittest.TestCase):
    def test_val_only_release_encodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = play(["e2e4", "e7e5"])
            manifest = write_teacher_release(root, {"train": [], "val": records})
            metadata = prepare_teacher_policy(manifest, root / "encoded", depth=1)
            self.assertEqual(metadata["rows_train"], 0)
            self.assertEqual(metadata["rows_val"], 2)
            data = TokenizedPositions(root / "encoded", "val", legal=True)
            ids, move, legal = data[0]
            self.assertEqual(decode_board(ids), (chess.STARTING_FEN, 1))
            self.assertEqual(move, UCI_MOVE_TO_IDX["e2e4"])
            self.assertEqual(len(legal), 20)


@unittest.skipUnless(shutil.which("stockfish"), "Requires Stockfish on PATH")
class TeacherDataTests(unittest.TestCase):
    def test_real_teacher_encoding_legality_and_source_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            board = chess.Board()
            train = [{"fen": board.fen(), "repetition": 2}]
            board.push_uci("e2e4")
            val = [{"fen": board.fen()}]
            release = {"format_version": 2, "files": {}}
            for split, records in [("train", train), ("val", val)]:
                path = root / f"{split}.jsonl"
                annotate_positions(records, path, AnnotationConfig(depth=2, multipv=2))
                with path.open("rb") as handle:
                    digest = file_digest(handle, "sha256").hexdigest()
                release["files"][split] = [
                    {"path": str(path), "sha256": digest, "rows": 1}
                ]
            manifest = root / "release.json"
            write_manifest(manifest, release)
            metadata = prepare_teacher_policy(manifest, root / "encoded", depth=2)
            self.assertEqual(metadata["rows_train"], 1)
            data = TokenizedPositions(root / "encoded", "train", legal=True)
            ids, move, legal = data[0]
            self.assertEqual(decode_board(ids), (chess.STARTING_FEN, 2))
            self.assertEqual(len(legal), 20)
            self.assertIn(move, legal)
            with self.assertRaises(FileExistsError):
                prepare_teacher_policy(manifest, root / "encoded", depth=2)
            # A checksum change must fail before publishing a dataset manifest.
            with (root / "train.jsonl").open("a") as handle:
                handle.write(json.dumps(train[0]) + "\n")
            with self.assertRaises(ValueError):
                prepare_teacher_policy(manifest, root / "bad", depth=2)
            self.assertFalse((root / "bad/manifest.json").exists())
            # A valid source file reused in both splits must fail overlap checks.
            release["files"]["train"] = release["files"]["val"]
            write_manifest(manifest, release)
            with self.assertRaises(ValueError):
                prepare_teacher_policy(manifest, root / "overlap", depth=2)
