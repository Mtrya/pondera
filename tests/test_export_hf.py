import json
import tempfile
import unittest

import pyarrow.parquet as pq
from test_release import CASES, build_run
from tools.export_hf import export_hf

from pondera.data.dataset import (
    HF_COLUMNS,
    HF_SCHEMA,
    LOCAL_ONLY_COLUMNS,
    TRACE_SCHEMA,
    load_manifest,
    write_manifest,
)
from pondera.data.pipeline import produce_shards, validate_release


class ExportTests(unittest.TestCase):
    def test_real_release_projection_preserves_shared_rows_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            config, roots, run, provider = build_run(directory, cases=CASES, seed=649)
            source_config = config.sources[0]
            for index, root in enumerate(roots):
                root["split"] = "val" if index == 0 else "train"
                root["source"] = {
                    "archive_url": source_config.url,
                    "archive_sha256": source_config.sha256,
                    "file": "lichess_elite_2025-11.pgn",
                    "game": 12460 + index,
                    "ply": 54,
                }
            release = run / "release"
            produce_shards(
                roots,
                release,
                config,
                sidecar_for_root=provider,
                shard_size=2,
                workers=1,
            )
            produce_shards(
                roots,
                release,
                config,
                sidecar_for_root=provider,
                shard_size=2,
                workers=1,
            )
            self.assertTrue(validate_release(release, roots, workers=1)["ok"])
            result = export_hf(release, run / "export", batch_size=1)
            self.assertEqual(result["rows"], {"train": 4, "validation": 1})

            local = []
            manifest = load_manifest(release / "release.json")
            for shard in manifest["shards"]:
                table = pq.read_table(release / shard["path"] / "data.parquet")
                self.assertEqual(table.schema, TRACE_SCHEMA)
                self.assertEqual(len(table.schema), 17)
                local.extend(table.to_pylist())
            for split, local_split in (("train", "train"), ("validation", "val")):
                path = run / "export" / "data" / f"{split}-00000.parquet"
                table = pq.read_table(path)
                self.assertEqual(table.schema, HF_SCHEMA)
                self.assertEqual(table.column_names, list(HF_COLUMNS))
                self.assertEqual(len(table.schema), 11)
                self.assertFalse(set(LOCAL_ONLY_COLUMNS) & set(table.schema.names))
                expected = [
                    {name: row[name] for name in HF_COLUMNS}
                    for row in local
                    if row["split"] == local_split
                ]
                self.assertEqual(table.to_pylist(), expected)
                for row in expected:
                    source = json.loads(row["source"])
                    self.assertEqual(source["archive_url"], source_config.url)
                    self.assertEqual(source["archive_sha256"], source_config.sha256)
                    self.assertEqual(
                        set(source),
                        {"archive_url", "archive_sha256", "file", "game", "ply"},
                    )

            rows = [row for row in local if row["split"] == "train"]
            self.assertEqual(
                {row["presentation"] for row in local}, {"zero", "full", "random"}
            )
            self.assertTrue(any(row["final_value"] is None for row in rows))
            self.assertTrue(any(row["mate_distance"] is not None for row in rows))
            self.assertTrue(any(" BOARD " in row["trace_text"] for row in rows))

            export_hf(release, run / "export-again", batch_size=1)
            for name in ("train-00000.parquet", "validation-00000.parquet"):
                self.assertEqual(
                    (run / "export" / "data" / name).read_bytes(),
                    (run / "export-again" / "data" / name).read_bytes(),
                )

            manifest_path = release / "release.json"
            original = load_manifest(manifest_path)
            for field in ("format_version", "grammar_version", "tokenizer_version"):
                with self.subTest(field=field):
                    write_manifest(manifest_path, {**original, field: -1})
                    with self.assertRaises(ValueError):
                        export_hf(release, run / f"invalid-{field}")


if __name__ == "__main__":
    unittest.main()
