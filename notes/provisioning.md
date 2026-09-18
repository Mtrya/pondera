# Provisioning — 初期准备决策与状态

记录已冻结的决策（D1–D7 + 命名 + schema）与初期准备清单的完成状态。更新规则：状态变化时改本文件；决策本身变更需项目所有者拍板。

## 冻结决策

| 编号 | 决策 | 内容 |
|---|---|---|
| 命名 | 项目名 | **pondera**（拉丁语 ponderāre，称量/沉思）。GitHub 仓库 Mtrya/chess-transformer → Mtrya/pondera |
| D1 | 对手 | Stockfish **17.1**，路径 `/usr/games/stockfish`（apt 安装），单线程，固定深度阶梯 d6/d8/d10/d12；冻结版本，所有棋力数字以此为基准 |
| D2 | 对局协议 | cutechess 系工具（实际选用 fastchess）；社区标准开局集；Syzygy 5 子 adjudication；单局上限 200 ply；我方基线一律 temperature=0、pondering=off |
| D3 | Schema 三层 | ① 数据集 schema（parquet: `fen, move, score?, depth?, count?, source, split`，版本化，变更必须迁移）② checkpoint manifest（`config.json`：arch/tokenizer/action_space/data 版本 + 推理默认 + 训练元信息）③ 对局边界 = UCI，运行器永不适配模型内部 |
| D4 | 代码工作流 | 短 PR 直接合 main；旧模型兼容层只保留到第一次评估冒烟完成 |
| D5 | 算力 | 8×A100 用 Slurm，资源到位时所有者通知；本地 RTX 3060 12GB = 开发+probe 机 |
| D6 | 实验跟踪 | swanlab；评估成绩进 git 管理的 leaderboard |
| D7 | OPD 合规性 | 在学生自己走到的状态上调 SF 打标（on-policy 蒸馏 / ExIt 形态）**不违反**"训练回路零搜索"：约束禁止的是搜索作为训练回路内的策略改进算子（如 MCTS 自对弈产生训练目标），SF 作为固定外部标注器与离线蒸馏同构。深入探索留待后续 |
| D8 | 两阶段结构 | ①**架构阶段**：判据 = 统一评估协议（SF 17.1 d1/d4/d8 ladder + 分难度 puzzle，两者同报）下的 SL 棋力；**数据与监督形式可随架构不同**（如轨迹消费型架构喂 SF 中间表示，不能只喂 state→move），但每臂记录数据构成、训练算力预算对齐。附计算弹性诊断（循环步数↔准确率，不参与排名）。②**RL 算法阶段**：在 Phase 1 shortlist 上开展，允许重开前二名架构复核。解耦理由：from-scratch RL 不可行、SL 先验必要；且解耦使实验数为加法而非乘法。SL target 可复合：best move/action-value + 搜索轨迹（UCI 级起，schema v2 扩展） |

## 硬件现状

- 本机：RTX 3060 12GB（`nvidia-smi` 确认），无 stockfish 之外的象棋工具；g++ 14.2 / cmake 3.31 可用。主机内存 37GB。
- **执行环境**：本 agent 的 shell 运行在 `kimi-bridge.service`（user manager，MemoryMax=4G）的 cgroup 里，且 cgroup 成员身份不随 setsid/nohup 改变——任何长时/重型任务（训练、批量评估、SF 标注）必须用 `systemd-run --user --unit=<name>` 启动到独立 unit，否则会被 cgroup OOM 静默 SIGKILL（见运维事件 2026-09-16）。linger 已开启，user unit 可存活过登出。
- HF 资产：`kaupane/chess-positions`（human_train 1,067,295,309 行 `fen,next_move,count`；stockfish_train 67,414,412 行，均无 score）；`kaupane/ChessFormer-SL` / `kaupane/ChessFormer-RL`（100.7M，仅作历史基线）。

## 初期准备清单

- [x] AGENTS.md（三节定稿）
- [x] 仓库卫生：删 `utils/` 死代码（所有者完成）、修 `chess_core/engine.py` `self.top_p` bug、`evaluate.py` 适配新数据 schema
- [x] 快速 tokenizer（`chess_core/tokenize.py`，FEN→uint8 张量）+ parity 校验（512 局面通过）
- [x] 评估基建：fastchess（`bin/fastchess`，gitignored）、UCI 适配器（`uci.py`）、对局运行器（`run_match.py`）、leaderboard（`results/leaderboard.md`）、旧 checkpoint 冒烟 ladder（d1/d4/d8 各 32 盘）
- [x] probe：HF stockfish_train 子采样 15M、离线 tokenize、policy-only 训练（2026-09-18 22:43 完成，29296 步；中途两次让卡暂停零损失；结果与读数见 `notes/handoff.md` 与 leaderboard。swanlab https://swanlab.cn/@mtrya/pondera ）
- [x] 文献浅调研（`notes/research/landscape.md`，52 篇条目；核心发现：Xiangqi 无搜索 RL 消融复现 v1 退化模式、ChessBench action-value 规模律、"循环步数 vs 稠密加深"是公开空白）
- [x] 代码侧改名（所有者完成：pyproject `pondera`、README、类名 PonderaModel/PonderaConfig）
- [x] PR 合 main（本 PR）；GitHub 仓库改名 Mtrya/chess-transformer → Mtrya/pondera 紧随合并执行（`gh repo rename`）

## 待办（依赖外部条件，暂不启动）

- [ ] 8×A100 到位后：RL 管线立项（等 probe + 文献结论）
- [ ] 带 score 的 SF 标注重跑（等 CPU 资源形态明确；teacher depth 必须高于当前评估档位）
- [ ] 开局集选型/构建（冒烟阶段可先用少量手工 EPD）
- [ ] Syzygy 5 子表下载（评估基建之后）
- [ ] 文献深度调研（触及实验设计时再做）

## 运维事件

### 2026-09-16 probe 训练被杀（14.7 小时进度损失）

- 现象：15:04 发现训练进程消失，日志停在 14:33:32（step 14800，epoch 2 第 152 步），stdout 无任何 traceback。
- 根因：训练是 `kimi-bridge.service` cgroup 的成员（Bash 工具启动的子进程，setsid/nohup 不改变 cgroup 归属），该 service 限额 MemoryMax=4G。epoch 2 开始重建 DataLoader 时新旧 worker 短暂叠加，cgroup 超限，14:33:54 被 systemd 按 OOM 规则 SIGKILL，整个 service（含训练与 bridge 本体）被停止重启。证据：`journalctl --user-unit=kimi-bridge.service` 的 "Killing process 131823 (pt_data_worker) with signal SIGKILL" 与 "3G memory peak, 5.7G memory swap peak"。
- 加重因素：`SAVE_EVERY_STEPS=20_000` 导致第一个周期 ckpt 还没写出（首个在 step 20000），无可 resume 点，14800 步全损。
- 处置：① 训练改由 `systemd-run --user --unit=pondera-probe --collect` 启动（独立 cgroup，MemoryMax=infinity），15:06 从 step 0 重启，step 1 loss 与上次逐位一致（同 seed 可复现），pos/s 288.6 已验证；② `SAVE_EVERY_STEPS` 20_000→2_000（最坏损失 2h）；③ checkpoint 改原子写（tmp + os.replace）。
- 教训：本机一切重型任务必须走 systemd-run（见"硬件现状-执行环境"）；长训练的首次 ckpt 间距必须 ≤ 可容忍损失时长。
