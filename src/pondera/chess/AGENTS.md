# chess/ 模块指南

棋域层：棋子、词表、编解码、镜像。它被数据编码与模型输入输出共同依赖，是序列化数据与 checkpoint 的契约层。

## 各文件职能

- `domain.py`：棋域原语——棋子与格子的编号表、重复计数。不识 token id。
- `vocabulary.py` + `vocabulary.txt`：词表的唯一权威。`vocabulary.txt` 一行一个 token 的规范名字，行号即 id；`vocabulary.py` 定义各段的算术布局常量，并在导入时校验文件与布局一致。词表的数值语义（value 分箱边界等）不在此文件。
- `codec.py`：FEN/重复计数与 token 数组之间的编解码、合法走法索引，以及 value/mate token 的数值语义（分箱边界、代表值、mate 覆盖）。
- `symmetry.py`：黑白镜像——token 镜像表、UCI 走法反射、走法置换表。

## 稳定性提示

本模块应当保持稳定。数据 manifest 与 checkpoint 通过 `CHESS_TOKENIZER_VERSION` 钉住词表布局；任何改动都会使既有编码数据与训练产物失效。确需改动词表时，用 `uv run python -m pondera.chess.vocabulary` 重新生成 `vocabulary.txt`，审阅 diff，并同步重编码数据。
