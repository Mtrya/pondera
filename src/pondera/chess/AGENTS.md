# chess/ 模块指南

棋域层：棋子、词表、编解码、镜像、路径跟踪、merge 字典。它被数据编码与模型输入输出共同依赖，是序列化数据与 checkpoint 的契约层。

## 各文件职能

- `domain.py`：棋域原语——棋子与格子的编号表、重复计数。不识 token id。
- `vocabulary.py` + `vocabulary.txt`：词表的唯一权威。`vocabulary.txt` 一行一个 token 的规范名字，行号即 id；`vocabulary.py` 定义各段的算术布局常量，在导入时校验文件与布局一致，并暴露分段与走法 id 映射（`SEGMENTS`、`UCI_MOVE_TO_IDX`、`IDX_TO_UCI_MOVE`）。词表的数值语义（value 分箱边界等）不在此文件。
- `codec.py`：FEN/重复计数与 token 数组之间的编解码、合法走法索引，以及 value/mate token 的数值语义（分箱边界、代表值、mate 覆盖）。
- `symmetry.py`：黑白镜像——token 镜像表、UCI 走法反射、走法置换表。
- `tracker.py`：纯规则路径状态机——根局面与两种重复口径（观察编码含根前历史，搜索终局只看已写路径）、绝对 UCI 路径、当前 python-chess 局面、合法走法与 BOARD 观测编码。不做评估与搜索，不 import data/models/training。
- `merge.py`：复合 token 层的纯序列函数——候选子串计数、按编码器实际用量生长的字典遴选、min-cost DP 编解码、merge-dictionary v1 TSV 读写。复合 id 从 `CHESS_VOCAB_SIZE` 起递增，PAD 保持最大原子 id；不 import data/。

## 稳定性提示

本模块应当保持稳定。数据 manifest 与 checkpoint 通过 `CHESS_TOKENIZER_VERSION` 钉住 board token 格式；trace 语法由 sequence 数据 manifest 的 `grammar_version` 单独钉住。任何改动都会使既有编码数据与训练产物失效。确需改动词表时，用 `uv run python -m pondera.chess.vocabulary` 重新生成 `vocabulary.txt`，审阅 diff，并同步重编码数据。
