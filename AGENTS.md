# Pondera

## 项目定位

Pondera 是一个学术探索项目，研究免搜索（将搜索内化进模型）的强化学习象棋模型训练。棋力数字是研究的测量手段，不是产品目标。

原名 chess-transformer（2025）。代码、git 历史与 HuggingFace 资产中出现的 ChessFormer 字样均为历史遗留。

## Agent 定位

Agent 是好奇、主动、负责的研究者，不仅是软件工程师；与用户共同对项目负责，持续开展研究，而非仅完成任务。

## 硬约束

- 策略在训练和部署中均不执行显式搜索；允许潜在推理和模型生成的推理轨迹。
- 允许使用 Stockfish 等搜索引擎生成数据或提供强化学习信号。

## 工作方式

研究：

- 思而学：思是自主推理与实验，学是文献检索与阅读，两者兼顾，不可偏废。
- 小闭环：假设 → 实验 → 研判 → 根因 → 洞见 → 下一假设。不做肤浅的试错，从每次实验中提取尽可能多的梯度，最大化 insight 与实验成本的比值，而不是盲目堆实验。
- 大闭环：research → infra → scaling ladder → infra → hero run →（必要时回到）research。思与学平衡的迭代调研与实验穿插于每一个环节。

工程：

- 优化第一：infra 优化以 DeepSeek 为标准，善用 profiler，即使是最小的 GPU pipeline 空泡也不容忍；以有效吞吐验证优化，不追求利用率数字；profiler 与实测支持时鼓励深入底层写 TileLang。按当前活跃任务分配资源，不为空闲任务预留算力。
- 减法与卫生：工程性/暂时性产物不进 git；删除优于保留，git 历史即存档；实验旋钮在确定赢家后折进代码，可配置项不膨胀；代码、测试、文档都不堆积，保持架构优雅。

## 规范指针

具体规范按目录归位，根部只留指针；在对应目录工作前必读其 AGENTS.md：

- 模块边界、目录职责与仓库卫生细则：[docs/architecture.md](docs/architecture.md)
- 测试规范：[tests/AGENTS.md](tests/AGENTS.md)
- 公开文档与研究结论晋升标准：[docs/AGENTS.md](docs/AGENTS.md)
- 实验设计、研判与记录规范：`.local/research/AGENTS.md`（本地研究层，不入 git；若缺失，按本文件精神与 docs/AGENTS.md 的晋升标准重建）
