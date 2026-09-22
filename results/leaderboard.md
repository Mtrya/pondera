# Leaderboard

Historical paired matches against Stockfish 17.1 at fixed depth. Pondera uses deterministic legal policy argmax with no search or pondering. Elo differences are relative to the specified opponent configuration; undefined confidence intervals are preserved as reported.

| date | checkpoint | opponent | games | W-D-L | score | elo_diff | 95% CI | notes |
|---|---|---|---|---|---|---|---|---|
| 2026-09-15 22:58 | kaupane/ChessFormer-SL | sf-d1 | 32 | 4-7-21 | 0.234 | -205.64 | ±113.77 | Historical SL baseline |
| 2026-09-15 22:59 | kaupane/ChessFormer-SL | sf-d4 | 32 | 0-6-26 | 0.094 | -394.11 | ±140.93 | Historical SL baseline |
| 2026-09-15 23:00 | kaupane/ChessFormer-SL | sf-d8 | 32 | 0-1-31 | 0.016 | -719.74 | ±nan | Historical SL baseline |
| 2026-09-18 22:44 | policy-ce-15m-2epochs | sf-d1 | 32 | 0-5-27 | 0.078 | -428.75 | ±170.86 | 15M positions, two epochs |
| 2026-09-18 22:45 | policy-ce-15m-2epochs | sf-d4 | 32 | 0-1-31 | 0.016 | -719.74 | ±nan | 15M positions, two epochs |
| 2026-09-18 22:45 | policy-ce-15m-2epochs | sf-d8 | 32 | 0-1-31 | 0.016 | -719.74 | ±nan | 15M positions, two epochs |
| 2026-09-18 22:45 | policy-ce-15m-step-20000 | sf-d4 | 32 | 0-1-31 | 0.016 | -719.74 | ±nan | 15M positions, step 20000 |
| 2026-09-18 22:49 | kaupane/ChessFormer-SL | sf-d1 | 32 | 4-7-21 | 0.234 | -205.64 | ±113.77 | Baseline reproduction |
| 2026-09-18 22:50 | kaupane/ChessFormer-SL | sf-d4 | 32 | 0-6-26 | 0.094 | -394.11 | ±140.93 | Baseline reproduction |
