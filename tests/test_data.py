import tempfile
import unittest
from pathlib import Path

from pondera.data.storage import load_manifest, write_manifest


class StorageTests(unittest.TestCase):
    def test_failed_manifest_write_preserves_previous_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            original = {"completed": [1, 2]}
            write_manifest(path, original)
            with self.assertRaises(TypeError):
                write_manifest(path, {"completed": [1, 2, 3], "invalid": object()})
            self.assertEqual(load_manifest(path), original)
            updated = {"completed": [1, 2, 3]}
            write_manifest(path, updated)
            self.assertEqual(load_manifest(path), updated)
