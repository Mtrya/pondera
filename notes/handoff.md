# Handoff — 2026-09-18（probe 收尾）

> 仓库目录改名后原 session 丢失，下一个 session 从本文件恢复上下文。
> 稳定决策见 `notes/provisioning.md`（D1–D8），项目宪法见 `AGENTS.md`。本文件只记"进行到哪一步"。

## 当前状态（截至 2026-09-18 深夜）

probe 实验完整闭环：训练完成 → ladder 评估 → v1 对照复现 → 结果入 leaderboard → 本分支经 PR 合并 main（状态见 git log）。
**下一步等所有者拍板 Phase 1 架构 shortlist**（先深调研，见"下一步"节）。监测 cron 已删除，无后台任务在跑。

## probe 实验最终结果（32 盘/档，SF 17.1 固定深度，确定性协议）

| checkpoint | d1 | d4 | d8 |
|---|---|---|---|
| probe-15M-2ep（final，29296 步） | 0.078（−429±171） | 0.016（−720，CI nan） | 0.016（−720） |
| probe-15M-mid（step 20000） | — | 0.016（−720） | — |
| v1 基线（ChessFormer-SL，9-15） | 0.234（−206±114） | 0.094（−394±141） | 0.016（−720） |
| v1 对照复现（9-18，与 9-15 逐位一致） | 0.234（−206） | 0.094（−394） | — |

读数：
- **probe 弱于 v1**，方向符合预期——15M 局面/2 epoch/policy-only CE 正是文献里最弱的监督配置（ChessBench：action-value > state-value > 行为克隆）。
- **d4/d8 在当前强度段饱和**（都是全败/单和，Elo 显示 −720±nan 是工具下限）。当前唯一有分辨率的档位是 d1；Phase 1 的模型比较需要分辨率方案（加盘数 / 加中间档 / 让子或 SF 限制档），不要拿饱和的 d4 当判据。
- probe 的实际棋局质量：对 SF d1 会出现漏防 Qxf2# 一类的初级失误、残局无价值概念乱动王——policy-only 无搜索的典型失败模式。

## 训练与工程状态

- probe 训练：15M 局面、2 epoch、policy-only CE；中途两次让卡暂停/恢复（step 6000、20000），零损失。训练经 `systemd-run --user --unit=pondera-probe` 运行（**本机一切重活都必须 systemd-run**，见 provisioning 运维事件：kimi-bridge cgroup 4G OOM 事故）。
- swanlab 接入：https://swanlab.cn/@mtrya/pondera （run id 存在 ckpt 里，resume 自动续曲线；指标 train/loss·lr·grad_norm·pos_per_s、val/top1·loss、gpu_mem）。
- `uci.py` lazy-load 修复：模型改到首次 `go` 时加载，握手即时应答（原实现会在 HF 缓存模型的元数据网络检查上拖垮 fastchess 启动超时——v1 对照曾因此起不来；修后对照逐位复现，推理语义不变）。
- 遗留小坑：`chess_core/engine.py` 的 `compute_repetition` 每次新建 Pool（慢，仅旧 Engine 路径）；`app.py`（Gradio demo）与 `evaluate.py` 的 `play_games`/`analyze_game_quality` 路径未实测。

## 下一步（按优先级）

1. **所有者拍板 Phase 1 架构 shortlist**。前置：深调研（pondering/循环架构、目标形状、表示设计），深调研触及实验设计时再做。
2. **H1 目标形状实验**：action-value 头（128 桶 + HL-Gauss）vs 现 policy CE，同数据同预算；判据按 D8（ladder + 分难度 puzzle 必须同报）。
3. **轨迹标注管线**（UCI 级：深度 × top-K × 分数）是轨迹消费型架构的前置依赖；数据 schema v2 与标注一起设计。
4. 8×A100 到位前不启动 RL 管线设计；已有 D8 的 Phase 2 结构约束。
5. 若某次 ladder 需要更高分辨率：优先加 d1 盘数与中间档，别用饱和档下结论。

## 环境备忘

- SF 17.1 在 `/usr/games/stockfish`；GPU RTX 3060 12GB；Python 用 `.venv/bin/python`。
- 训练停止/重启：`systemctl --user stop pondera-probe` / `systemd-run --user --unit=pondera-probe --collect -p WorkingDirectory=... -p StandardOutput=append:... <venv python> train_probe.py --resume <ckpt>`（完整命令见 provisioning）。
- HF 模型缓存：`~/.cache/huggingface`（v1 模型）；训练数据缓存：`data/probe/hf_home`。HF 匿名拉取有限速，缓存命中时优先考虑 `HF_HUB_OFFLINE=1`。
- `.gitignore` 的 `*.md` 白名单只放行 README/AGENTS/notes/**/results/**；另有 `data/probe/`、`bin/`、`swanlog/`、`config.json`。
- git 提交身份沿用历史：Mtrya <erchanmion20@outlook.com>（本仓库未设 config，用 `git -c` 传入）。
