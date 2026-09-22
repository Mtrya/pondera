import json
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import torch

from pondera.chess.mapping import PIECE_TO_IDX, UCI_MOVE_TO_IDX
from pondera.chess.position import repetition_count
from pondera.chess.tokenize import TOKENIZER_VERSION, encode_fen, encode_fens
from pondera.inference.policy import pick_move
from pondera.models.checkpoint import load_model, read_checkpoint, save_checkpoint
from pondera.models.transformer import PonderaModel

CONFIG = {"num_blocks": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2}
torch.set_num_threads(1)


class ModelTests(unittest.TestCase):
    def test_encoding_special_fields(self):
        board = chess.Board()
        board.push_uci("e2e4")
        tokens = encode_fen(board.fen(en_passant="fen"), 2)
        self.assertEqual(tokens.shape, (73,))
        self.assertEqual(tokens.dtype, np.uint8)
        self.assertEqual(int(tokens[chess.E4]), PIECE_TO_IDX["P"])
        self.assertEqual(int(tokens[64]), 1)
        np.testing.assert_array_equal(tokens[65:69], [1, 1, 1, 1])
        self.assertEqual(int(tokens[69]), chess.E3)
        self.assertEqual(int(tokens[72]), 1)
        self.assertEqual(len(UCI_MOVE_TO_IDX), 1969)
        self.assertIn("a7a8n", UCI_MOVE_TO_IDX)

    def test_history_is_preserved(self):
        board = chess.Board()
        for move in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
            board.push_uci(move)
        history = list(board.move_stack)
        self.assertEqual(repetition_count(board), 3)
        self.assertEqual(board.move_stack, history)

    def test_checkpoint_roundtrip_and_fen_interface(self):
        model = PonderaModel(**CONFIG).eval()
        fens = [chess.STARTING_FEN]
        ids = torch.from_numpy(encode_fens(fens))
        expected = model(ids=ids)
        for actual, reference in zip(
            model(fens, torch.ones(1, dtype=torch.long)), expected
        ):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            save_checkpoint(path, model, CONFIG)
            self.assertEqual(
                read_checkpoint(path)["tokenizer_version"], TOKENIZER_VERSION
            )
            restored = load_model(path)
            for actual, reference in zip(restored(ids=ids), expected):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            self.assertEqual(
                pick_move(restored, chess.Board()), pick_move(model, chess.Board())
            )

    def test_legacy_checkpoint_configuration(self):
        model = PonderaModel(**CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pth"
            for config_key in ("config", "model_config"):
                with self.subTest(config_key=config_key):
                    torch.save(
                        {config_key: CONFIG, "model_state_dict": model.state_dict()},
                        path,
                    )
                    loaded = load_model(path)
                    for key, value in loaded.state_dict().items():
                        torch.testing.assert_close(value, model.state_dict()[key])

    def test_checkpoint_tokenizer_version_is_required_and_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pth"
            save_checkpoint(path, PonderaModel(**CONFIG), CONFIG)
            checkpoint = read_checkpoint(path)
            for version in (None, TOKENIZER_VERSION + 1):
                with self.subTest(version=version):
                    invalid = dict(checkpoint)
                    if version is None:
                        del invalid["tokenizer_version"]
                    else:
                        invalid["tokenizer_version"] = version
                    torch.save(invalid, path)
                    with self.assertRaises(ValueError):
                        load_model(path)

    def test_hub_directory_roundtrip(self):
        model = PonderaModel(**CONFIG).eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = load_model(Path(directory))
            for key, value in loaded.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[key])
            config_path = Path(directory) / "config.json"
            config = json.loads(config_path.read_text())
            self.assertEqual(config["tokenizer_version"], TOKENIZER_VERSION)
            config["tokenizer_version"] = TOKENIZER_VERSION + 1
            config_path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_model(Path(directory))
            del config["tokenizer_version"]
            config_path.write_text(json.dumps(config))
            loaded = load_model(Path(directory))
            for key, value in loaded.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[key])

    def test_policy_masks_illegal_moves(self):
        model = PonderaModel(**CONFIG).eval()
        with torch.no_grad():
            model.act_proj.weight.zero_()
            model.act_proj.bias.zero_()
            model.act_proj.bias[UCI_MOVE_TO_IDX["a7a8q"]] = 100
            model.act_proj.bias[UCI_MOVE_TO_IDX["e2e4"]] = 10
        self.assertEqual(pick_move(model, chess.Board()), "e2e4")
        mate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        self.assertEqual(pick_move(model, mate), "0000")

    def test_missing_checkpoint_and_invalid_batch(self):
        with self.assertRaises(FileNotFoundError):
            load_model(Path("/nonexistent/model.pth"))
        with self.assertRaises(ValueError):
            encode_fens([chess.STARTING_FEN], [])
