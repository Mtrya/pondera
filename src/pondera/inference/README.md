# Model inference

`trace.py` loads the checkpoint's architecture, weights, and merge dictionary, then uses the common autoregressive `model(input_ids, cu_seqlens)` interface. Generation recomputes the full prefix at each step. Adding a model architecture requires registering its constructor in `MODEL_CLASSES`; generation and parsing have no architecture-specific implementation.

A reasoning budget counts model calls before `FINAL`. Natural `FINAL` starts the answer early; reaching the reasoning limit inserts `FINAL` without a model call. A zero budget prefills `FINAL`. The separate answer allowance counts model tokens after that delimiter. Answer generation is greedy and unconstrained: no field, move, value, PV, or `END` is repaired or masked. A premature `END` or exhausted answer budget produces a failed answer.

`parse_answer` expands dictionary tokens into base atoms before interpreting the suffix. Equivalent segmentations, including composites crossing answer-field boundaries, produce identical parsed answers. The prefix before `FINAL` is opaque. A complete suffix contains a decision, a value, optional mate information, zero or more continuation moves, and `END`. PV continuation excludes the already-selected decision. Moves are interpreted in the root player's token frame and replayed in absolute coordinates; legal draw claims require the supplied repetition count or halfmove clock. Format validity, decision legality, PV legality, and failure codes remain separate.

`vanilla.py` retains the legal-masked, zero-presentation move policy used by `vanilla_uci.py` and existing match callers. It shares checkpoint loading with `trace.py`. `uci.py` owns the model-independent UCI loop. The unconstrained checkpoint evaluator uses `trace.py` directly.
