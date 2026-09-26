# Board attribute experiments

These configurations investigate how piece identity and location enter a fixed-depth policy. Each uses the same 262,144-position depth-16 training release, the same 8,192 validation positions, effective batch 512, seed 640 and a four-pass, 2,048-update warmup/cosine schedule. Their 28.5M-parameter models retain eight layers and width 512. They are controlled exploratory interventions; a short-run ranking does not settle representation or scaling potential.

| Configuration | Intervention | Parameters |
|---|---|---|
| `absolute.toml` | Additive piece and absolute-square embeddings; unmasked policy CE | 28,498,866 |
| `joint.toml` | One embedding per piece–square pair, including empties; FFN width 1504 | 28,492,210 |
| `oriented.toml` | Additive inputs oriented to the side to move; output logits restored to absolute move coordinates | 28,498,866 |
| `legal.toml` | Additive absolute inputs; normalize the training loss over legal moves | 28,498,866 |

The joint representation has 832 board entries plus the existing metadata entries. Reducing FFN width keeps total parameters within 0.024% of the reference; report the resulting theoretical FLOPs separately. The original version-2 discrete data remains lossless: the embedding maps each piece and square to its joint index without changing the stored dataset or action vocabulary.

Player-relative inputs reflect ranks and swap piece colors when Black is to move. Castling fields and en-passant coordinates undergo the same transformation. The original side flag remains available, so the supplied position is exactly recoverable; counters and repetition metadata remain exact. Model outputs use absolute action indices, allowing identical labels, legality masks and evaluation rows across candidates. This is a change of coordinates, not a symmetry constraint forcing identical predictions for color-swapped positions.

The legal-loss intervention asks whether supplying legality during training changes the learning of preferences among legal alternatives. It does not add search. Always report both raw and legal-normalized metrics, including rank metrics and calibration probes. Compare each intervention with the matching reference; do not attribute a difference between joint embeddings and legal loss to tokenization alone.

The shorter schedule is motivated by observed generalization and calibration deterioration under repeated fitting of limited data. It differs from the twenty-pass schedule in `../exposure/`; use the new reference for within-study conclusions. Evaluate/checkpoint every 256 updates, preserve per-position predictions, and analyze learning trajectories rather than select by final CE. Keep dynamic geometry and other lossless encodings open. Any promising representation needs a common-data scale check before adoption.

For a unique-data intervention, compare the reference's four passes over 262,144 positions with two passes over a nested 524,288-position release, using the same teacher, validation set, total 1,048,576 presentations, 2,048 updates and LR schedule. This holds exposure and theoretical training FLOPs fixed while changing data diversity. Additional data may subsequently support representation comparisons, but must be frozen and shared within each comparison.
