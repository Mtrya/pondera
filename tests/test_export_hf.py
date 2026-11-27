import json
import tempfile
import unittest

import pyarrow as pa
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
                path = run / "export" / "data" / f"{split}-00000000.parquet"
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
                    self.assertEqual(
                        set(source),
                        {"archive_url", "file", "game", "ply"},
                    )

            rows = [row for row in local if row["split"] == "train"]
            self.assertEqual(
                {row["presentation"] for row in local}, {"zero", "full", "random"}
            )
            self.assertTrue(any(row["final_value"] is None for row in rows))
            self.assertTrue(any(row["mate_distance"] is not None for row in rows))
            self.assertTrue(any(" BOARD " in row["trace_text"] for row in rows))

            export_hf(release, run / "export-again", batch_size=1)
            for name in ("train-00000000.parquet", "validation-00000000.parquet"):
                self.assertEqual(
                    (run / "export" / "data" / name).read_bytes(),
                    (run / "export-again" / "data" / name).read_bytes(),
                )

            for cap in (1, 2500):
                parts = run / f"parts-{cap}"
                result = export_hf(release, parts, batch_size=2, max_part_bytes=cap)
                self.assertGreater(len(result["files"]), 2)
                for item in result["files"]:
                    self.assertTrue(item["rows"] == 1 or item["bytes"] <= cap)
                for split, label in (("train", "train"), ("validation", "val")):
                    found = [
                        row
                        for path in sorted((parts / "data").glob(f"{split}-*.parquet"))
                        for row in pq.read_table(path).to_pylist()
                    ]
                    self.assertEqual(
                        found,
                        [
                            {name: row[name] for name in HF_COLUMNS}
                            for row in local
                            if row["split"] == label
                        ],
                    )
                repeated = run / f"parts-again-{cap}"
                export_hf(release, repeated, batch_size=3, max_part_bytes=cap)
                for item in result["files"]:
                    self.assertEqual(
                        (parts / item["path"]).read_bytes(),
                        (repeated / item["path"]).read_bytes(),
                    )

            manifest_path = release / "release.json"
            original = load_manifest(manifest_path)
            for field in ("format_version", "grammar_version"):
                with self.subTest(field=field):
                    write_manifest(manifest_path, {**original, field: -1})
                    with self.assertRaises(ValueError):
                        export_hf(release, run / f"invalid-{field}")

            write_manifest(manifest_path, {**original, "tokenizer_version": -1})
            self.assertEqual(
                export_hf(release, run / "text-only")["rows"],
                {"train": 4, "validation": 1},
            )

    def test_part_sizing_uses_compressed_bytes_across_row_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            config, roots, run, provider = build_run(
                directory,
                cases=("start_d4",),
                recipe_text="[recipe]\npresentations = {full = 1.0}",
            )
            release = run / "release"
            produce_shards(roots, release, config, sidecar_for_root=provider, workers=1)
            path = release / "shard-000000" / "data.parquet"
            (original,) = pq.read_table(path).to_pylist()
            rows = 4096
            records = [
                {
                    **original,
                    "root_id": str(index),
                    "source": json.dumps({"game": index}),
                }
                for index in range(rows)
            ]
            table = pa.Table.from_pylist(records, schema=TRACE_SCHEMA)
            pq.write_table(table, path)
            manifest_path = release / "release.json"
            manifest = load_manifest(manifest_path)
            manifest["shards"][0]["rows"] = rows
            write_manifest(manifest_path, manifest)

            cap = 256 * 1024
            self.assertGreater(table.select(HF_COLUMNS).nbytes, cap)
            output = run / "compressed"
            result = export_hf(release, output, max_part_bytes=cap)
            (part,) = result["files"]
            self.assertLessEqual(part["bytes"], cap)
            with pq.ParquetFile(output / part["path"]) as exported:
                self.assertEqual(exported.num_row_groups, 4)
                self.assertEqual(
                    exported.read().to_pylist(), table.select(HF_COLUMNS).to_pylist()
                )


if __name__ == "__main__":
    unittest.main()
