# 文献地形扫描（Landscape Scan）

- 日期：2026-09-15
- 方法：WebSearch + FetchURL（arXiv 摘要/HTML 优先）。**广而浅**：每篇只写一段判断，不做单篇深挖；数字只写能从原文/摘要核实的。
- 目的：为 Pondera（免搜索训练 + 搜索内化的闭环研究）确定"已知边界在哪、空白在哪、下一步先验证什么"。
- 覆盖：①免搜索/弱搜索 RL 的证据边界 ②内化搜索的机制设计空间 ③象棋表示与目标设计 ④蒸馏规模律 ⑤评估方法论；另附 2024–2026 新工作索引与 "recurrent depth / looping" 直接证据。
- 读法：每条固定四行——核心结论 / 证据强度 / 对我们的含义 / 可对应的验证实验。`证据强度` 是我对"这条结论能不能撑起设计决策"的判断，不是论文自己的措辞。

---

## 一、免搜索/弱搜索 RL 在棋类上的证据边界

### Mastering Chess and Shogi by Self-Play（AlphaZero, 2017）
链接：https://arxiv.org/abs/1712.01815
- 核心结论：自对弈 + MCTS + 策略/价值网络，从随机权重出发达到超人棋力；搜索既生成训练目标也参与自我改进。
- 证据强度：极强（多组独立复现：Leela Chess Zero、ELF OpenGo、KataGo）。
- 对我们的含义：定义了对照组。它证明"搜索在训练回路里"能走多远，也说明我们的核心问题应是"去掉它以后损失发生在哪一环"。
- 可对应实验：用 lc0 的网络做 1-node 前向（policy-only）对比多节点搜索，量化"搜索→单次前向"的 Elo 损耗；这是评估阶梯的噪声基线，也校准我们 ladder 的分辨率。

### Grandmaster-Level Chess Without Search / Amortized Planning with Large-Scale Transformers（Ruoss et al., 2024）
链接：https://arxiv.org/abs/2402.04494
- 核心结论：纯监督蒸馏 Stockfish 16 的 action-value（10M 局 → 5.3 亿局面、153 亿个 (局面, 走法) 标注），最大 270M 参数 transformer 不做任何搜索即达 Lichess 快棋 Elo 2895（对人类）；action-value 目标 > state-value > 行为克隆，128 桶分类 + HL-Gauss 平滑优于回归/交叉熵。
- 证据强度：强（NeurIPS 2024，开放数据集与代码；但外部 Elo 只跑了 174 局，内部联赛 2299 与外部分数 2895 差约 600 Elo，说明锚点不同）。
- 对我们的含义：**这是我们 SL 目标函数的直接替代方案**——我们现在是 policy 交叉熵 + 标量 value；他们把"每个合法走法的胜率分布"作为唯一监督，policy 由 argmax 派生。
- 可对应实验：3060 上的 probe 加一个 action-value 头（K=128 桶 + HL-Gauss），与现有 policy/value 双头做同数据同步数的 puzzle 准确率 + 对 SF17.1 d6 的 ladder 对比。

### Mastering Chinese Chess AI (Xiangqi) Without Search（Chen & Shu, 2024）
链接：https://arxiv.org/abs/2410.04865
- 核心结论：SL（人类高手 + alpha-beta 标注，分层采样）→ 不搜索的 PPO 自对弈，达人类天梯前 0.1%；四条消融：同参数量 ViT > ResNet（对基线胜率 38.6% → 65.1%）；把"双方可行走法"作为输入特征显著加速训练；**动态对手池（DOP）优于纯自对弈（84% → 92.5%）**；**标准 GAE（γ=λ=1，全轨迹 bootstrap 到终局）会训崩（胜率 65% → ~10%），改成有截断的 VECT 后升到 92.5%**。
- 证据强度：中强（单一团队/单一棋种，但消融完整、失败模式写得具体；未见独立复现）。
- 对我们的含义：v1 RL 阶段"从 SL 退化"最可能的机制就是 value/advantage 估计问题；DOP 与 VECT 是两个可直接照抄的实验变量，且都在"零搜索"前提下验证过。
- 可对应实验：在 SL checkpoint 上做 3×2 消融：{纯自对弈, 对手池} × {全轨迹 GAE, L 步截断 GAE}，用 fastchess 对 SF17.1 d6/d8 测 200+ 局，看是否复现"崩溃 vs 稳定"。

### Suphx: Mastering Mahjong with Deep RL（2020）
链接：https://arxiv.org/abs/2003.13590
- 核心结论：无搜索（纯策略梯度自对弈）在不完美信息多玩家游戏上超过 99.99% 人类；三个关键稳定化技术：全局奖励预测 GRP（把稀疏终局奖励变成密集预测信号）、oracle guiding（用上帝视角模型早期引导）、run-time policy adaptation（推理期在线适应）。
- 证据强度：强（Tenhou 天凤平台大规模真人排名，历史首次）。
- 对我们的含义：麻将比象棋更接近"长轨迹 + 稀疏奖励"，它的结论说明：**免搜索 RL 的可行路径是把终局奖励换成学出来的密集 value 信号**，而不是靠更长训练。
- 可对应实验：RL 阶段加一条"终局结果的 n-step 截断 value 目标"（而非只用 z），对比只用 z 的版本；观察 value loss 与对 SF 胜率曲线是否解耦。

### The Surprising Effectiveness of Approximate Value Iteration in Self-Play（2026）
链接：https://arxiv.org/abs/2609.09094
- 核心结论：在四子棋、7×7 Hex 与合成博弈上，最朴素的近似价值迭代（AVI）+ 一步贪心策略，学到的 value 比 AlphaZero 更准，策略以更低的训练/推理代价与 MCTS 策略相当；在 Othello 与 9×9 Go 上初步稳定。
- 证据强度：中（有真值 oracle 可精确评估，但规模小、未到象棋级）。
- 对我们的含义："搜索的收益可能被高估"这一方向值得保留：如果我们的价值函数足够准，一步贪心的强度上限可能被低估。
- 可对应实验：SL 阶段同时训练 value 头，评估"policy argmax" vs "对合法走法 value argmax"（一步前瞻）两种出招的棋力差。

### Prior-Directed Exploration for Searchless Chess（2026）
链接：https://arxiv.org/abs/2608.27757
- 核心结论：对已蒸馏 MCTS 访问次数的无搜索网络做单遍 RL 微调：把熵正则换成朝网络自身 MCTS 先验的**前向（mass-covering）KL**，并用 value 头的不确定性自适应采样温度；约 2000 步把 puzzle 准确率 93.9% → 94.9%、mate-in-4 77% → 81%，但**棋力只在基线上方轻微浮动**；只在 puzzle 上微调的对照组战术提升最大却掉约 260 Elo。
- 证据强度：中（单文、单一基座、规模小；但"准确率与棋力解耦"的对照设计很干净）。
- 对我们的含义：**这是最贴近我们课题的 2026 工作**——无搜索 RL 微调确实能稳定地改战术指标，但棋力增益不是自动的；puzzle 指标不能当作棋力代理。
- 可对应实验：把 puzzle accuracy 与 ladder Elo 作为两个必须同时报告的指标；任何 RL 变体若只涨前者不涨后者，按"未通过"处理。

### Policy improvement by planning with Gumbel（Danihelka et al., 2022, ICLR）
链接：https://openreview.net/forum?id=bERaNdoegnO
- 核心结论：Gumbel AlphaZero/MuZero 在小模拟预算下保证策略改进；少量模拟即可接近大量模拟的效果。
- 证据强度：强（ICLR spotlight，已被 MiniZero、EfficientZero V2 等复现）。
- 对我们的含义："弱搜索"是一个独立实验轴：如果研究需要搜索，应优先选"少模拟但有改进保证"的形式，而不是把 MCTS 全量塞回训练。
- 可对应实验：若未来把搜索作为推理期配置，先实现 2–16 模拟的 Gumbel 根选择，测其相对 policy-only 的 Elo 增量曲线（作为"搜索预算→棋力"标定）。

### MiniZero: Comparative Analysis of AlphaZero and MuZero（2024）
链接：https://arxiv.org/abs/2310.11305
- 核心结论：在同一代码基上系统比较 AlphaZero/MuZero 家族（含 Gumbel 变体）在 Go、Othello、Atari 上的表现，给出算法选择与预算的经验规律。
- 证据强度：中强（开源、可复现的多环境对照）。
- 对我们的含义：若日后做搜索消融，这是"别自己造对照"的现成参照与实现清单。
- 可对应实验：不需要立刻做；在我们确定推理期搜索配置时作为选型参考。

### AlphaStar: Grandmaster level in StarCraft II（Vinyals et al., 2019）
链接：https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/
- 核心结论：不依赖搜索的 PPO 智能体在复杂实时博弈达宗师级；关键工程是 league training（主智能体 + 利用者/主利用者、多智能体联盟）解决非传递策略循环。
- 证据强度：强（Nature，人类天梯前 0.2%）。
- 对我们的含义：对手池/联盟是免搜索 RL 稳定性的产业级答案；象棋的非传递性虽弱于 RTS，但同一起点（SL ckpt）的自对弈仍易塌到单一线路。
- 可对应实验：对手池的最小实现：保留最近 N 个 checkpoint + 按当前胜率倒数采样（Xiangqi 的 exp(-r/τ) 形式），先验证"是否阻止 policy 熵塌/线路塌缩"。

### A Unified Game-Theoretic Approach to Multiagent RL (PSRO, 2017) / Neural Fictitious Self-Play（2016）
链接：https://arxiv.org/abs/1711.00832 ；https://arxiv.org/abs/1603.01121
- 核心结论：自对弈的替代理论框架：用"策略种群 + 响应 oracle + 元策略"逼近纳什，NFSP 则让神经网络直接学平均策略（fictitious play）。
- 证据强度：强（理论 + 后续大量工作与综述）。
- 对我们的含义：我们的"对手池"本质上是 PSRO/NFSP 的轻量近似；若 RL 长期不稳，这里有现成的做法与失败诊断语言。
- 可对应实验：记录 RL 期间与各历史 checkpoint 的胜率矩阵，用元策略求解器（即使只做简单均匀/最优混合）检查是否存在非传递循环。

### Value targets in off-policy AlphaZero: a new greedy backup（Willemsen et al., 2021）
链接：https://ir.cwi.nl/pub/30870/30870.pdf
- 核心结论：系统梳理 AlphaZero 系 value target 的选择（终局结果、MCTS 值、n 步 bootstrap、贪心 backup），指出不同目标对稳定性与强度的差异。
- 证据强度：中（分析 + 实验型论文，非顶级会议，但问题定义清晰）。
- 对我们的含义：直接回答"value bootstrapping vs 纯终局奖励"——在回合制棋类里，纯终局 z 的方差与长轨迹目标不一致是经典痛点；MuZero 在棋盘游戏里对 value 直接 bootstrap 到终局（≈预测最终结果）。
- 可对应实验：RL 阶段把 value 目标分别设为 {终局 z, 截断 n 步 bootstrap, MCTS/搜索值}（第三项仅作离线对照），比较 value loss 曲线与 Elo。

### Can Large Language Models Develop Strategic Reasoning? Post-training Insights from Learning Chess（2025）
链接：https://arxiv.org/abs/2507.00726
- 核心结论：用棋类做 RLVR（可验证奖励）的受控实验：稠密的"走法质量"奖励（来自棋类预训练的 action-value 网络）通常优于稀疏二值胜负奖励，但**所有模型都停在远低于专家的水平**；作者认为瓶颈在预训练模型对象棋的内部理解不足，RL 难以弥补。
- 证据强度：中（LLM 设定，与我们的 scaled policy 不同，但消融规范）。
- 对我们的含义：提醒"奖励变稠密 ≠ 能力涌现"——RL 的上限受底座表征质量约束；先把 SL 做强，再谈 RL。
- 可对应实验：RL 前先量化底座指标（puzzle 准确率、对 SF 的 top-k 一致率、value 校准），把这些指标与 RL 后的 Elo 增量做相关性分析（n≈个位数起步，作为长期档案）。

---

## 二、「内化搜索」的机制设计空间

### （重看）Ruoss et al. 2024：容量 vs 计算步数说了什么
链接：https://arxiv.org/abs/2402.04494
- 核心结论：容量维度上，9M/136M/270M 的内部联赛 Elo 为 2025/2259/2299，puzzle 88.9%/94.5%/95.4%——**在所有指标上"更大就更好"，且在 10M 局、153 亿标注下仍未过拟合**；计算维度上，他们比较的是外部搜索：同代 Lc0 网络 policy-only 2292 vs 400 模拟 2858（+566 Elo）；作者在局限性中明确写道：Lc0 用自对弈 RL 训练的网络在"不搜索"时也比他们的 SL transformer 更强。
- 证据强度：强（数据、代码、多组消融公开）；但"计算步数"轴他们只做了外部对比，**没有做循环前向/多步精化**的对照。
- 对我们的含义：他们的结论不能被读成"容量比计算步数重要"，只能说"在他们的设定里，容量可扩展且尚未饱和，而搜索的价值无法被单次前向吸收"。这正是我们要补的空白。
- 可对应实验：固定参数量，比较 {K 倍宽的稠密模型} vs {同一 backbone 循环 K 次}：同 FLOPs 下比 puzzle 准确率与 ladder Elo——这是"容量 vs 计算步数"在无搜索象棋上的第一次直接对照（我们可负担的最小版本在 3060 上可行）。

### Mastering Board Games by External and Internal Planning with Language Models（Schultz et al., ICML 2025）
链接：https://arxiv.org/abs/2412.12119
- 核心结论：在象棋/Chess960/四子棋/Hex 上，LM 既可作为 MCTS 的先验（外部搜索），也可被训练成"在上下文里生成一棵线性化搜索树并给出最终选择"（内部搜索/内化搜索）；后者把搜索行为蒸馏进前向过程，达到特级大师级水平且更接近人类搜索预算。
- 证据强度：强（ICML 2025，DeepMind，多游戏 + 对抗 SOTA 引擎）。
- 对我们的含义：**"把搜索写进权重"这一目标的直接先例**，并且给出了可操作配方：先训领域知识（transition/value），再训"生成搜索轨迹"；其内化对象是文本化的树，而我们可以用 latent/recurrent 步骤作为载体。
- 可对应实验：以 SF 的浅层搜索轨迹（例如 depth 6 的 PV 行）为监督，训练模型预测"思考步骤"再输出走法；与纯走法监督对照，比 puzzle 与 ladder。

### Evidence of Learned Look-Ahead in a Chess-Playing Neural Network（Jenner et al., 2024）
链接：https://arxiv.org/abs/2406.00877
- 核心结论：Lc0 的策略网络在单次前向中已内含"前瞻"：未来走法所在格子的激活对输出有因果影响；存在把信息"向前/向后跨时间"搬运的注意力头；线性探针能以 92% 准确率预测两步后的最优走法。
- 证据强度：中强（机制可解释性标准做法：激活干预 + 探针；只在"单一最佳线路"的局面成立）。
- 对我们的含义：**"无搜索网络内部已经在做类似搜索的计算"的最硬证据**；也提示我们可以在自己的模型上做同类探针，直接观察内化程度。
- 可对应实验：在我们的 checkpoint 上训练 linear probe 预测 K 步后的 SF 首选走法，把探针准确率作为"内化深度"指标，跨 SL 数据量/循环步数对比。

### Understanding the learned look-ahead behavior of chess neural networks（2025）
链接：https://arxiv.org/abs/2505.21552
- 核心结论：对上述现象的细化：前瞻行为高度**依赖局面上下文**（不是普遍机制），需在特定类型的局面上才出现。
- 证据强度：中（对前作的独立复核与细化）。
- 对我们的含义：内化搜索可能是"局部能力"而非全局性质——用单一探针数字下结论要谨慎；也提示训练分布会决定内化发生在哪些局面类型上。
- 可对应实验：把探针准确率按局面类型（战术/残局/开局）分层报告，而不是只报一个总数。

### Thinking Fast and Slow with Deep Learning and Tree Search（ExIt, 2017）
链接：https://arxiv.org/abs/1705.08439
- 核心结论：Expert Iteration：交替"用搜索产生更好的示范"与"把示范蒸馏回策略网络"，即搜索→策略的内化循环；在 Hex 上大幅超越纯策略学习。
- 证据强度：强（奠基性工作，被 AlphaZero 系与后续搜索蒸馏工作反复引用）。
- 对我们的含义：**内化搜索的最简可检验形式**：训练回路零搜索的硬约束并不禁止我们把"离线搜索产物"当作监督目标（需与项目约定区分：训练回路的梯度/数据生成是否允许离线搜索标签——我们当前的 SL 蒸馏本质就是 ExIt 的 M 步）。
- 可对应实验：ExIt 轮次实验：SL → 用浅搜索对自身局面重标注 → 再蒸馏，迭代 2–3 轮，看 ladder 是否单调上升（这是"把搜索搬进权重"的最短路径）。

### Training Large Language Models to Reason in a Continuous Latent Space（Coconut, 2024）
链接：https://arxiv.org/abs/2412.06769
- 核心结论：把中间推理步骤放在连续隐空间（最后一层隐状态回灌）而非语言空间，在需要回溯的规划任务上优于 CoT 且更省 token。
- 证据强度：中强（ICLR 2025，多个逻辑推理基准；后续复现工作多，出现"概念瓶颈/中间事实被覆盖"等批评）。
- 对我们的含义：latent CoT 是把"思考步骤"从文本接口解放出来的模板；对我们的意义是"用额外的前向步骤承担计算"而非"输出思考文本"。
- 可对应实验：在无搜索象棋里做最小 latent-CoT：在输出走法前插入 K 个可学习 latent 步（沿残差流回灌），K=1/2/4 对比同 FLOPs 的稠密加深。

### Think before you speak: pause tokens（2023）
链接：https://arxiv.org/abs/2310.02226
- 核心结论：在输入中插入可学习的 [PAUSE] token（推理时也插）能提供额外计算，报告在推理/QA 任务上有一致提升；效果依赖训练配方与规模，并非普遍成立。
- 证据强度：中（实验结果可复现，但增益在后期微调/不同规模下会削弱）。
- 对我们的含义：与我们的 FEN + readout token 设计最接近——已经有两个 learnable readout token；"ponder token"可视为把 readout 数量/深度参数化。
- 可对应实验：把 readout token 数量从 2 提到 K（K=4, 8），单独观察棋力变化，作为"额外前向计算是否被利用"的最低成本探针。

### Let's Think Dot by Dot: Hidden Computation in Transformer LMs（2024）
链接：https://arxiv.org/abs/2404.15758
- 核心结论：无意义填充 token（如 "."）能让 transformer 解决需要串行计算的任务，但需要特定监督形式；揭示"额外的 token 位置 = 隐式计算位"。
- 证据强度：中（算法任务上的受控实验，规模小）。
- 对我们的含义：对我们的棋盘 token 设计有直接含义——序列长度/占位符本身是计算预算；不需要真"想"的内容，也可能带来能力提升。
- 可对应实验：在 FEN 序列后追加 K 个占位 token（不做任何语义约束），看是否为模型带来可测量的棋力增量（相对同参数不加的对照）。

### Quiet-STaR（2024）
链接：https://arxiv.org/abs/2403.09629
- 核心结论：让模型在每个 token 前生成内部 rationale，并用对未来文本的预测奖励来筛选，从而在无任务微调的情况下提升零样本推理。
- 证据强度：中强（有确定性收益，但成本高、后续复现门槛高）。
- 对我们的含义：给出了"如何用自监督奖励塑造内部思考"的机制；对我们的价值在于——思考步骤可以被**奖励信号**塑造，而不只是被监督模仿。
- 可对应实验：若引入 latent 步骤，用"该步骤能否提升对最终结果（胜/负、SF 一致性）的预测"作为辅助奖励（类似 GRP 的精神）。

### Adaptive Computation Time（2016）/ PonderNet（2021）/ Universal Transformers（2018）
链接：https://arxiv.org/abs/1603.08983 ；https://arxiv.org/abs/2107.05407 ；https://arxiv.org/abs/1807.03819
- 核心结论：ACT 引入每步 halting 概率与 ponder cost；PonderNet 把它变成可学习的概率式停机；Universal Transformer 用权重共享的循环深度 + ACT，在算法与语言任务上优于同深度非共享模型。
- 证据强度：强（这三篇是自适应计算/循环深度的共同祖先，被后续所有 looped/ponder 工作引用）。
- 对我们的含义："ponder"（沉思）在本项目里的技术对应物就是这三条：循环块 + 停机策略 + 计算代价正则；这是我们要实现的目标形态。
- 可对应实验：最小实现 = transformer 块权重共享 + K 步固定循环（先不学停机），只回答一个问题：**同为 100M 参数，循环 4 步是否比 4 倍深的稠密网络更强或更省**。

### Confident Adaptive Language Modeling（CALM, 2022）
链接：https://arxiv.org/abs/2207.07061
- 核心结论：用置信度阈值做 token 级早退，在不损质量的前提下减少解码计算；给出可靠的质量—计算权衡框架。
- 证据强度：中强（有理论保证 + 大规模实验）。
- 对我们的含义：早退是"自适应深度"的另一条实现路径，但对棋类这种"每步都需要完整判断"的任务，早退的收益点可能在残局/简单局面（即 ponder 的反面：什么时候不必想）。
- 可对应实验：分析我们模型在局面难度（SF 评分差、唯一最佳走法与否）上的策略熵分布，先确认是否存在"简单局面占多数"的可省计算空间。

### Mixture-of-Depths（2024）/ Mixture-of-Recursions（2025）
链接：https://arxiv.org/abs/2404.02258 ；https://arxiv.org/abs/2507.10524
- 核心结论：MoD 让每个 token 动态决定是否参加某层的计算（静态 FLOP 预算下等价于动态深度）；MoR 把递归深度做成 token 级可学习路由，报告同精度下显著降低训练/推理 FLOPs。
- 证据强度：中强（语言模型上的规模实验，工程细节敏感）。
- 对我们的含义：给出"哪里值得多想"的路由机制；棋盘 token 天然有 spatial 结构（关键格子 vs 无关格子），MoD 式路由可能比全局加深更划算。
- 可对应实验：对 1969 维动作空间做 token 级路由并不现实；更可行的是"局面级"路由（用一个 gate 决定本局面循环几步），可作为 ponder 的粗粒度版本先跑通。

### Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach（Huginn, 2025）
链接：https://arxiv.org/abs/2502.05171
- 核心结论：循环深度 transformer 在测试时可展开任意步数，把"隐空间推理"变成第三种测试时计算轴；随循环步数增加，数学/推理任务性能提升。
- 证据强度：中强（3.5B 规模、公开复现，但收益依赖训练时的随机深度采样技巧）。
- 对我们的含义：**最直接的"计算步数可换能力"证据之一**；其关键工程点是训练时随机化循环步数（类似随机深度），否则模型不会在不同步数下都工作。
- 可对应实验：实现循环块时，训练期对 K 做随机采样（1..K_max），评估期扫 K，看 ladder 是否随 K 单调——若否，说明我们的架构没有把额外计算用起来。

### Scaling Latent Reasoning via Looped Language Models（Ouro, 2025）
链接：https://arxiv.org/abs/2510.25741
- 核心结论：7.7T token 预训练的循环 LM，1.4B/2.6B 可匹配 4B/8B 标准 transformer；优势来自"知识操纵/迭代精化"能力而非知识存储。
- 证据强度：中强（大预算预训练 + 多基准；预印本阶段）。
- 对我们的含义：循环深度在"参数效率"上的最有力证据（2–3× 参数等价）；如果我们的 pondering 架构成立，应当看到类似的参数效率差。
- 可对应实验：与 H1 共用实验：小参数量 + 多循环 vs 大参数量单遍，在同等 FLOPs 下比较（本机 3060 只做 puzzle/小 ladder；A100 到位后复现大规模点）。

### Reasoning with Latent Thoughts: On the Power of Looped Transformers（Saunshi et al., ICLR 2025）/ Looped Transformers as Programmable Computers（2023）
链接：https://arxiv.org/abs/2502.17416 ；https://arxiv.org/abs/2301.13196
- 核心结论：理论上证明循环 transformer 的表达力可模拟多步推理（latent thoughts）；在等 FLOP 对比下，循环模型的困惑度更差但推理类任务更好——即"循环给的归纳偏置偏向推理而非记忆"。
- 证据强度：强（理论 + 受控实验）。
- 对我们的含义：这正是我们想要的偏置：象棋需要的是搜索/推理，不是记忆开局库；但也要接受"同 FLOPs 下语言建模指标更差"的代价。
- 可对应实验：在 puzzle 集（考推理）与开局库记忆集（考记忆）上分别评估，验证循环模型是否"推理增益 > 记忆损失"。

### Hierarchical Reasoning Model（HRM, 2025）/ Tiny Recursive Model（TRM, 2025）
链接：https://arxiv.org/abs/2506.21734 ；https://arxiv.org/abs/2510.04871
- 核心结论：两个小网络以不同频率递归 + 深监督，用约 27M 参数、约 1000 条样本在数独/迷宫/ARC-AGI 上取得强结果；TRM 用更简单结构超过 HRM。
- 证据强度：中（结果亮眼但争议大：ARC Prize 分析与"Critical Supplementary Material"指出其测试时训练/集成等因素对结论影响很大；数据效率结论不等于棋力结论）。
- 对我们的含义：如果棋盘任务真的需要"迭代精化"，小模型 + 多次循环的原则性证据在这里；但不能照搬结论，只借鉴结构（快/慢两个模块、深监督）。
- 可对应实验：把快/慢模块改成"局面编码器 + 循环精化块"，在 puzzle 上做同参数量对照；同时按 ARC Prize 式的严格协议（固定步数、禁测试时训练）评估，避免自欺。

### MuZero（2019/2020）/ EfficientZero（2021）/ UniZero（2024）
链接：https://arxiv.org/abs/1911.08265 ；https://arxiv.org/abs/2111.00210 ；https://arxiv.org/abs/2406.10667
- 核心结论：学到的隐动力学模型（representation/dynamics/prediction）配合树搜索，在棋盘游戏与 Atari 上达到强水平；EfficientZero 用自监督一致性损失大幅提升样本效率；UniZero 把世界模型做成可扩展的 transformer 隐空间规划。
- 证据强度：强（MuZero/EfficientZero 被广泛复现；UniZero 为中强）。
- 对我们的含义："学习动力学"是内化搜索的另一条路线：模型可以学会自己滚动推演（rollout）而不是显式建树；对应我们可能的"模型内部 rollout K 步"辅助任务。
- 可对应实验：加一个动力学辅助头：给定局面 + 走法预测下一局面的 value（或直接预测 K 步后局面表征），看是否提升长战术 puzzle 表现。

### Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters（Snell et al., 2024）
链接：https://arxiv.org/abs/2408.03314
- 核心结论：在中等难度问题上，给小模型更多推理计算可以超过参数量大 14 倍的模型；在极难问题上计算替代规模的收益有限。
- 证据强度：中强（有影响力的经验研究，后续大量引用与复现）。
- 对我们的含义：为"容量 vs 计算步数"提供外部先例与预期形状：优势应在中等难度局面（而非最深的战术题）上最明显。
- 可对应实验：把 puzzle 按 Lichess 难度分档，报告各档上"循环步数增量"与"参数量增量"的收益比，验证是否复现"中等难度占优"的形状。

### Looped World Models（2026）/ Understanding Dynamic Compute Allocation in Recurrent Transformers（2026）
链接：https://arxiv.org/abs/2606.18208 ；https://arxiv.org/abs/2602.08864
- 核心结论：前者把循环 transformer 用于环境模拟与动力学推演；后者分析循环 transformer 中动态计算分配（学到 vs 坍塌）的行为，指出自适应分配不总按预期工作。
- 证据强度：中（新工作，尚未沉淀；但方向与我们的 ponder 设计直接相关）。
- 对我们的含义：循环结构做"世界模型"是一个正在成形的方向；同时后者提醒：自适应计算可能退化成固定步数，"学到停机"要单独验证。
- 可对应实验：在固定 K 循环跑通后，记录不同局面下循环步数的收益差（按局面难度分层），先确认"自适应"是否有可学信号，再引入停机门。

---

## 三、象棋专属的表示与目标设计

### Mastering Chess with a Transformer Model（Chessformer, Monroe & Chalmers, 2024）
链接：https://arxiv.org/abs/2409.12272
- 核心结论：关键不在堆参数，而在**位置表示是否进入注意力机制**：可表达的位置编码让模型以 8× 更少计算超过 AlphaZero 的棋力与解题能力，以 30× 更少计算匹配此前特级大师级 transformer；并展现出传统引擎不擅长的局面理解（受困子力、堡垒）。
- 证据强度：强（同时被 Lc0 项目工程化采用；作者与 Lc0 团队有重叠）。
- 对我们的含义：我们当前 FEN token 序列 + 学习式位置编码正是"弱表示"的一侧；棋盘几何结构（行列、相对位移）应当显式进入注意力偏置。
- 可对应实验：在现有 100M 模型上做最小消融：给注意力加相对棋盘位置偏置（绝对/相对坐标），比较同预算下的 puzzle 准确率（低成本、可直接跑）。

### Chessformer: A Unified Architecture for Chess Modeling（2026）
链接：https://arxiv.org/abs/2605.19091
- 核心结论：encoder-only + 方格 token + 新的几何注意力偏置（Geometric Attention Bias）+ **基于注意力的 from-to 策略头**，同时推进三件事：人类走法预测 57.1%（参数量不到前 SOTA 的 1/4）、集成进 Lc0 后 +100 Elo 并战胜 Stockfish、注意力可直接归因到棋盘格（可解释性）。
- 证据强度：中强（2026 年新工作；Lc0 集成与比赛结果为其背书）。
- 对我们的含义：**from-to 结构化的策略头比 1969 维扁平动作空间更贴合象棋动作几何**；我们的动作空间是枚举 1969 个合法走法，可考虑改造为"起点×终点"双因子分解（Xiangqi 论文也用了 16×90 的分解）。
- 可对应实验：在 3060 上比较 {1969 维扁平头} vs {64×64 from-to 头（掩码非法）} 的收敛速度与 puzzle 准确率——低成本、结论明确。

### Distilling Stockfish with One Billion Positions（Gigafish, 2026, 博客）
链接：https://blog.lukesalamone.com/posts/distilling-stockfish/
- 核心结论：39 亿局面（固定 depth 10 标注）数据集上训练 10 亿样本；纯 ViT 显著弱于 ResNet 与 "ResNet→Transformer 混合"，混合架构最优；训练全程 loss 持续下降，作者据此认为**数据量仍是瓶颈**；并批评 ChessBench 是变深度标注、且 state-value 仅 5.3 亿个。
- 证据强度：中（个人博客 + 数据集发布，非同行评审；但方法透明、可复现）。
- 对我们的含义：对我们直接影响的有两点：①固定 teacher 深度 ②平面（bitboard）输入 + 卷积归纳偏置对我们的 token 序列设计是反向证据——如果我们坚持 token 序列，应当用位置偏置补偿几何归纳偏置（见 Chessformer 条目）。
- 可对应实验：在 SF 标注子集上做 A/B：token 序列 vs bitboard 平面（小模型即可），看数据效率差异；这直接影响我们后续是否换输入格式。

### Accelerating Self-Play Learning in Go（KataGo, 2019/2020）
链接：https://arxiv.org/abs/1902.10565 ；https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md
- 核心结论：辅助目标与自对弈机制设计带来约 1.4 GPU-年达到 ELF 的水平（约 50× 计算节省）；关键技巧：playout cap randomization、policy target pruning、global pooling、辅助策略目标（预测对手下一手）、**辅助 ownership/score 目标**；消融显示移除 ownership+score 会明显降低学习效率（各消融加速因子相乘约 9.1×）；作者给出通用启发："当目标可分解为子事件时，预测子事件通常有帮助"。
- 证据强度：强（开源、可复现、被广泛引用；消融为短程 run，加速因子是近似值）。
- 对我们的含义：**这是"辅助目标设计"最好的模板**，且它作用在自对弈（非蒸馏）场景——与我们的 RL 阶段直接相关；对应象棋的候选辅助目标：子力差、王安全、终局距离/剩余步数、对手最佳回应。
- 可对应实验：RL/SL 阶段加入"对手最佳回应预测"辅助头 + "剩余步数/终局距离"回归头，与不加对照比 Elo 与样本效率（KataGo 的经验是最短 2 天 run 才看得清差异，我们需要按数据量而非步数对齐比较）。

### Out-of-distribution Tests Reveal Compositionality in Chess Transformers（2025）
链接：https://arxiv.org/abs/2510.20783
- 核心结论：270M 象棋 transformer 具备组合泛化：在高度 OOD 局面仍几乎总是走合法着（规则外推），OOD puzzle 质量高；Chess960 上具备基础策略适应但弱于显式搜索的符号方法；训练早期先学会"只动自己的子"，暗示组合式理解是涌现的。
- 证据强度：中强（系统的 OOD 测试套件；单一模型规模）。
- 对我们的含义：合法性与规则泛化不是瓶颈；**瓶颈在需要搜索的战术深度**——这为我们把研究重心放在"计算步数/内化搜索"而非"表示学习补齐规则"提供了依据。
- 可对应实验：把该文的 OOD 测试集作为我们 probe 的固定评估项之一（合法性、OOD puzzle、Chess960 适配），作为"不是记忆"的证据。

### Emergent World Models and Latent Variable Estimation in Chess-Playing LMs（Karvonen, 2024）/ Othello-GPT（Li et al., 2023）
链接：https://arxiv.org/abs/2403.15498 ；https://arxiv.org/abs/2210.13382
- 核心结论：仅在走法序列上训练的自回归模型会自发形成棋盘状态的世界模型（Othello 的线性探针可解码棋格并可因果干预；象棋 LM 中存在类似的隐状态估计）；对 Othello-GPT 的后续批评指出其可能学的是"启发式集合"而非单一世界模型。
- 证据强度：中（探针/干预证据强；"世界模型"解释有争议，需要区分"可解码"与"被使用"）。
- 对我们的含义：走法/FEN 序列模型确实会构建隐式棋盘表征——这是我们"token 序列可行"的正面证据；但"可解码 ≠ 被用于决策"，评估时要小心。
- 可对应实验：对我们的模型做线性探针（子力、王位置、被攻击格），验证内部棋盘表征的完整度，并作为跨架构（序列 vs 平面）比较的中间指标。

### Stop Regressing: Training Value Functions via Classification for Scalable Deep RL（2024）
链接：https://arxiv.org/abs/2403.03950
- 核心结论：把 value 回归换成分类（1-Hot / HL-Gauss）能显著改善大规模 RL 的可扩展性；作者在象棋蒸馏任务上复现了"分类 > 回归"，并报告 HL-Gauss 更好。
- 证据强度：强（多领域实验 + 与 Ruoss 结论一致）。
- 对我们的含义：我们 v1 的 value loss 是标量回归（0.04 量级）——在稀疏胜负目标下，分类式分布 value 头更稳；这同时是 RL 稳定化与 SL 目标设计的一步棋。
- 可对应实验：value 头改成 101 桶分类 + HL-Gauss，SL 阶段测 value 校准（Brier/ECE）与 puzzle 表现，RL 阶段测 value loss 波动与 ELo。

---

## 四、蒸馏的规模律：局面数量/标签质量 → 棋力

### ChessBench 的规模消融（Ruoss et al., 2024）
链接：https://arxiv.org/abs/2402.04494
- 核心结论：10M 局（5.3 亿局面 / 153 亿 action-value，约等于 8864 天未并行化的 SF 计算）下，9M/136M/270M 参数模型在所有指标上单调提升且未过拟合；当训练集只有 10K 局时，≥7M 参数即开始过拟合；数据量增加对每个模型规模都带来提升。
- 证据强度：强（公开 dataset + 明确的 scaling 曲线）。
- 对我们的含义：**我们 HF 上 stockfish_train 有 6740 万行但无 score**；若只做行为克隆（只有 best move），会缺 action-value 的排序信息——这正是论文里"行为克隆最弱"的原因之一。数据量本身（千万级局面起）不是最紧的瓶颈，标签形状才是。
- 可对应实验：probe 设计成三臂：(a) 只有 best move（现有数据）(b) best move + 浅搜索 score（新增标注）(c) 全合法走法 action-value（大批标注，A100 期）；用同一 ladder 比较。

### Gigafish 的经验（2026, 博客）
链接：https://blog.lukesalamone.com/posts/distilling-stockfish/
- 核心结论：39 亿局面（固定 depth 10）→ 训练 10 亿样本，loss 全程下降、方向准确率 92.97%、残差 MAE 64cp，作者判断仍有大量数据可吃；同时指出 ChessBench 的 state-value 只有约 5.3 亿个且深度不固定。
- 证据强度：中（非同行评审，但流程透明、数据公开）。
- 对我们的含义：如果只做 state-value（每局面一个标量），我们需要的规模比 action-value 小 1–2 个数量级；**这提示一条性价比路线：先做大规模 state-value 蒸馏 + 一步 value-argmax 出招**（见 AVI 条目）。
- 可对应实验：用我们现有 SF 标注能力做 5000 万–1 亿局面的 state-value 数据集（固定 depth），训练 value-only 模型，与 policy 蒸馏对比 ladder；直接回答"我们需要多少标注"。

### Maia（2020）/ Maia-2（2024）
链接：https://arxiv.org/abs/2006.01855 ；https://arxiv.org/abs/2409.20553
- 核心结论：Maia 用约 1200 万局人类对局做行为克隆（按分段训练），走法匹配率约 50%（同期 SF 约 37%、Leela 约 42%）；Maia-2 统一为单模型 + 技能条件注意力，比 Maia 提升约 2 个百分点。
- 证据强度：强（微软研究院系列工作，数据与模型公开；但目标是模仿人类不是最强棋力）。
- 对我们的含义：**行为克隆的天花板由"人类走法分布"决定**，不能直接外推到棋力；但它是"每级数据量 → 匹配率"的少见公开曲线，可用于估算我们 SL 的边际收益。
- 可对应实验：在 probe 上同时报告"走法匹配率（对 SF 首选）"与"ladder Elo"，画出我们对数据量的曲线，与 Maia 的匹配率-数据量关系做对照。

### Learning Models of Individual Behavior in Chess（2020）
链接：https://arxiv.org/abs/2008.10086
- 核心结论：个性化微调中，玩家对局数从 1K → 40K 时，走法匹配率约 0.53 → 0.58，边际收益递减但持续。
- 证据强度：中（小规模个性化研究；曲线形状有参考价值）。
- 对我们的含义：给"数据量 → 质量"提供了另一条经验斜率（≈ +5pt / 40× 数据），可用于估算我们 10–20M 子采样 probe 的预期位置。
- 可对应实验：我们的 probe 用 5M/10M/20M/40M 四档，画匹配率-数据量曲线，验证斜率是否同量级。

### Scaling Scaling Laws with Board Games（Jones, 2021）/ MARL 规模律（Neumann & Gros, 2022）/ AlphaZero 神经规模律与 Zipf（2024）
链接：https://arxiv.org/abs/2104.03113 ；https://arxiv.org/abs/2210.00849 ；https://arxiv.org/abs/2412.11979
- 核心结论：AlphaZero 系智能体的 Elo 随训练计算呈对数线性增长（Hex），系数跨棋盘规模稳定；MARL 设定下也观察到 Elo 与参数/计算幂律关系（但大模型的训练难度会破坏幂律）；损失（尤其 value loss）不是棋力的可靠预测器。
- 证据强度：中强（多条独立证据线一致；具体系数依赖实现）。
- 对我们的含义：**"loss 下降 ≠ 棋力上升"** 必须写进我们的实验纪律：v1 就出现过 RL loss 变差、棋力未知的状况；凡是棋力结论必须来自 ladder。
- 可对应实验：probe 训练全程记录 {train loss, value loss, puzzle 准确率, 周期性 ladder Elo}，报告四者相关性——为后续大规模训练建立"能预测棋力的中间指标"。

### Leela Chess Zero 的工程实践
链接：https://lczero.org/dev/wiki/technical-explanation-of-leela-chess-zero/
- 核心结论：分布式自对弈 + 网络规模逐步放大（随算力升级网络），是"规模与数据并进"的产业级实践；训练数据是搜索增强的自对弈局面。
- 证据强度：中（工程文档与社区记录，非论文；但规模与成果真实）。
- 对我们的含义：他们的数据来自"搜索增强的自对弈"——正是我们被约束不能做的事；他们的强度位置（不搜索时 ~2292–2418 Elo，Ruoss 表）是我们 SL+RL 路线的现实参照。
- 可对应实验：把 lc0 的不同代网络（含小网络）纳入我们的评估对手池，作为外部标尺，避免只用 SF 一个锚点。

---

## 五、评估方法论

### Statistical Methods in Fishtest
链接：https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html
- 核心结论：Fishtest 用**五分模型（pentanomial）**而非三分（胜/和/负）建模对局结果，可在同置信度下节省大量测试资源；用 **GSPRT**（未知参数以 MLE 代入的广义序贯检验）做早停；测试边界用"归一化 Elo"表达，使预期测试时长不依赖和棋率与开局库；pentanomial 与 trinomial 的差异可用于估计**开局库偏差的 RMS**。
- 证据强度：强（Stockfish 生产系统 + 公式文档 + 模拟验证）。
- 对我们的含义：我们要建"社区标准"的评估管线，直接照这套做即可：以成对（同开局双色）比赛统计构造 pentanomial，用 GSPRT 做早停，报告归一化 Elo 与开局库偏差。
- 可对应实验：实现评估报告器：输入 PGN → 输出 trinomial/pentanomial Elo、CI、开局库偏差 RMS；用 v1 checkpoint 与 SF17.1 d6 的既有对局先验证工具链。

### SPRT（Chess Programming Wiki）
链接：https://www.chessprogramming.org/Sequential_Probability_Ratio_Test
- 核心结论：SPRT 需要四个参数（elo0, elo1, alpha, beta），LLR 触碰边界即停止；实践中应指定极大局数上限避免过早停止；**强引擎/弱引擎应使用不同边界**（Elo 压缩），社区惯例：Stockfish LTC [0.5,2.5]、STC [0,2]、Top30 [0,3]、其余 [0,5]/[0,10]；并给出开局库选择建议（弱引擎用均衡库如 8moves_v3，强引擎用偏置库如 UHO_Lichess 降低和棋率）。
- 证据强度：强（社区标准做法，Stockfish/CuteChess/OpenBench 全部遵循）。
- 对我们的含义：我们的 Elo 远低于 Stockfish 系，属"其余引擎"档 → 用 [0,10] 级别的宽松边界做早期迭代；但涉及关键结论（如 RL 是否变强）时应使用更小边界并配合 200+ 局固定局数。
- 可对应实验：为我们的 ladder 定义 3 档边界模板（粗筛 [0,10]、验证 [0,5]、结论 [-5,0]/[0,5]），固化进评估脚本。

### jw1912/SPRT
链接：https://github.com/jw1912/SPRT
- 核心结论：给出 SPRT 与 **GSPRT**（Fishtest 使用）在象棋语境下的推导与计算脚本，包含与 CuteChess/OpenBench 一致的口径说明。
- 证据强度：中（社区实现与文档，非论文；但与 fishtest 公式一致）。
- 对我们的含义：自己实现统计时用它做交叉验证，避免"自己发明一套统计"的常见坑。
- 可对应实验：对同一批对局，用 (a) 我们的实现 (b) 该脚本 (c) fastchess 内建 SPRT 三方对齐。

### fastchess / cutechess-cli
链接：https://github.com/Disservin/fastchess/blob/master/man.md ；https://www.chessprogramming.org/Engine_Testing
- 核心结论：fastchess 是新一代对局管理工具（支持 pentanomial 统计、SPRT、UCI 最小接口要求），cutechess-cli 是长期事实标准；两者支持开局库、并发、Syzygy adjudication 等。
- 证据强度：强（工程工具，实际生产使用）。
- 对我们的含义：我们已定 D2（fastchess + 社区开局集 + Syzygy 5 子 + 200 ply 上限）；这里补充的是"开局库要随强度选择"以及"UCI 接口最小集合"。
- 可对应实验：冒烟 ladder = 我们的 SL ckpt vs SF17.1 d6/d8（固定开局、双色、200 局），先测通工具链与 CI 宽度。

### BayesElo / Ordo / Whole-History Rating（2008）
链接：https://inria.hal.science/inria-00323349/document ；https://ijccrl.com/bayeselo-and-ordo-in-computer-chess/
- 核心结论：SPRT 只回答"两组之间是否有差"，**多引擎/多档位 ladder 需要贝叶斯或最大似然评分系统**：BayesElo 建模颜色优势与和棋、Ordo 做池内拟合、WHR 支持时间变化强度；所有 Elo 都是相对量，必须有锚点。
- 证据强度：强（Coulom 的 WHR 论文 + BayesElo/Ordo 长期作为 CCRL/KCEC 等主流榜单的标准工具）。
- 对我们的含义：我们的阶梯（我方多档 vs SF17.1 多档）本质是一个小型联赛 → 用 BayesElo/Ordo 拟合，并把最强锚点（如 SF d12）固定。
- 可对应实验：用 Ruoss 的做法锚定：先测我们模型对某个固定对手的绝对分，再把内部相对 Elo 平移过去；记录锚点来源与局数。

### 关于 Elo 的分辨率与和棋饱和
链接：https://beuke.org/chess-engine-draws/ ；https://arxiv.org/pdf/2105.00839
- 核心结论：引擎间和棋率随 Elo 上升而饱和（拟合显示约 3900 Elo 时达 99% 和棋），高分段分辨所需局数急剧增长；小规模联赛的 Elo 存在几十点的系统不确定度（Elo 本人给出的 15 轮比赛 probable error ≈ 49 点）。
- 证据强度：中强（数据拟合 + 历史文献 + 统计推导）。
- 对我们的含义：我们在 1500–2500 档位测 10–20 Elo 的差异需要成百上千局；且**不能把 vs-bots 与 vs-humans 的 Elo 混着报**（Ruoss 的 270M 模型 vs bots 2299、vs humans 2895，差约 600 Elo，同一模型！）。
- 可对应实验：对每个 ladder 结论同时报告：局数、和棋率、CI（bootstrap 或 BayesElo 输出）、对手池构成；<100 Elo 的结论在 <200 局时不作数。

### Human-aligned Chess with a Bit of Search（Allie, 2024）
链接：https://arxiv.org/abs/2410.03893
- 核心结论：人类对齐棋力评估需要"人类思考时间匹配"的搜索预算分配（ALLIE-ADAPTIVE-SEARCH 按预测的人类思考时间线性分配搜索量），否则与人类的对局数字不可比。
- 证据强度：中强（NeurIPS 系工作，含人类对局实验）。
- 对我们的含义：如果我们将来与人类或 Lichess 做外部校准，必须声明"单遍前向 vs 固定时间/节点"的预算设定；否则数字无意义。
- 可对应实验：在我们 ladder 报告模板中固定声明"我方 temperature=0、无 pondering、每步固定单次前向"（与 D2 一致），并把它写进 leaderboard 表头。

### 外部一致性检查：Ruoss 的评估实践（2024）
链接：https://arxiv.org/abs/2402.04494
- 核心结论：内部联赛 400 局/配对、ECO 开局、BayesElo（CI=0.5）并**锚定到 Lichess**；但 30–553 局的样本量给出 ±15–23 Elo 的区间，且 vs-bots 与 vs-humans 两个池子严重不一致。
- 证据强度：强（作为"坑"的实证案例，比方法论文章更有说服力）。
- 对我们的含义：给我们两个直接纪律：①锚点固定且公开 ②对手池构成必须记录，因为"pool 不同 → Elo 差几百"。
- 可对应实验：我们 leaderboard 的每一行强制带上 {锚点, 局数, 对手池, 开局集, 时间控制, 表内 CI} 六列。

---

## 六、2024–2026 年 chess + transformer + RL / reasoning 新工作索引

没有搜到名为 "Chess-R1" 的论文；与它最接近的是象棋（Xiangqi）RLVR 与"用棋做 RL 训练场"的一批工作。以下为一次性索引（标题 / 年份 / 链接 / 一句话定位）：

- **Amortized Planning with Large-Scale Transformers: A Case Study on Chess**（2024）https://arxiv.org/abs/2402.04494 —— 无搜索 SL 蒸馏 + 规模消融（本扫描的核心参照）。
- **Mastering Chess with a Transformer Model（Chessformer）**（2024）https://arxiv.org/abs/2409.12272 —— 位置表示/注意力设计换算力。
- **Mastering Chinese Chess AI (Xiangqi) Without Search**（2024）https://arxiv.org/abs/2410.04865 —— 最完整的"无搜索 SL→RL"消融（DOP/VECT/特征）。
- **Mastering Board Games by External and Internal Planning with LMs**（2024/ICML2025）https://arxiv.org/abs/2412.12119 —— 把搜索"内化"进 LM 上下文。
- **Evidence of Learned Look-Ahead in a Chess-Playing Neural Network**（2024）https://arxiv.org/abs/2406.00877 —— 单次前向已有前瞻的机制证据。
- **Understanding the learned look-ahead behavior of chess neural networks**（2025）https://arxiv.org/abs/2505.21552 —— 上述现象依赖局面上下文。
- **Out-of-distribution Tests Reveal Compositionality in Chess Transformers**（2025）https://arxiv.org/abs/2510.20783 —— 规则泛化强、搜索型战术弱。
- **Can LLMs Develop Strategic Reasoning? Post-training Insights from Learning Chess**（2025）https://arxiv.org/abs/2507.00726 —— 棋上 RLVR 的边界：稠密奖励更好但普遍平台期。
- **Xiangqi-R1**（2025）https://arxiv.org/abs/2507.12215 —— RL 提升象棋走法合法性与分析准确率（LLM 设定，510 万局面 SFT + 专家引导 RL）。
- **ChessArena**（2025）https://arxiv.org/abs/2509.24239 —— LLM 对弈/策略推理的评测平台（含 thinking 模型对比）。
- **LLM Chess**（2025）https://arxiv.org/abs/2512.01992 —— 50+ 模型的棋类推理与指令遵循基准（Elo + 每步诊断）。
- **Grounded Chess Reasoning in Language Models via Master Distillation**（2026）https://arxiv.org/abs/2603.20510 —— 用大师数据把语言推理"落地"到棋盘。
- **How Reasoning Evolves from Post-Training Data: An Empirical Study Using Chess**（2026）https://arxiv.org/abs/2604.05134 —— SFT→RL 中推理忠实性演化；直接预测最佳走法的 SFT 带来最强 RL，但推理不忠实。
- **Chessformer: A Unified Architecture for Chess Modeling**（2026）https://arxiv.org/abs/2605.19091 —— 几何注意力偏置 + from-to 策略头 + Lc0 集成（+100 Elo）。
- **Prior-Directed Exploration for Searchless Chess**（2026）https://arxiv.org/abs/2608.27757 —— 无搜索 RL 微调：前向 KL 探索、准确率与棋力解耦。
- **The Surprising Effectiveness of Approximate Value Iteration in Self-Play**（2026）https://arxiv.org/abs/2609.09094 —— 不用搜索的 AVI 与 MCTS 竞争（小游戏）。
- **Engine-Agnostic Search with Human Policy Guidance**（2026）https://arxiv.org/abs/2606.25176 —— 索引项：讨论引擎搜索与人类策略引导的三种技术路线（未细读）。
- **Looped World Models**（2026）https://arxiv.org/abs/2606.18208 —— 循环 transformer 做环境模拟/动力学。

---

## 七、「recurrent depth / looping 提升棋类或推理任务」的直接证据

- 语言/推理侧证据较多且一致：Huginn（测试时循环步数↑ → 推理任务↑，https://arxiv.org/abs/2502.05171）、Ouro（1.4B/2.6B 循环 LM ≈ 4B/8B 稠密，2–3× 参数效率，https://arxiv.org/abs/2510.25741）、Saunshi et al.（等 FLOP 下循环模型推理更好、困惑度更差，https://arxiv.org/abs/2502.17416）、HRM/TRM（小模型 + 递归 ≈ 小样本强解题，但复现争议大，https://arxiv.org/abs/2506.21734 、https://arxiv.org/abs/2510.04871）、Coconut / pause token / filler token（隐空间或占位符承担计算，https://arxiv.org/abs/2412.06769 、https://arxiv.org/abs/2310.02226 、https://arxiv.org/abs/2404.15758）。
- 自适应计算侧的证据更谨慎：ACT/PonderNet/Universal Transformer 提供机制（https://arxiv.org/abs/1603.08983 、https://arxiv.org/abs/2107.05407 、https://arxiv.org/abs/1807.03819），MoD/MoR 提供 token 级路由（https://arxiv.org/abs/2404.02258 、https://arxiv.org/abs/2507.10524），但 2026 年的分析指出循环 transformer 的动态计算分配常退化为固定步数（https://arxiv.org/abs/2602.08864）。
- **棋类侧：没有找到"循环深度 vs 棋力"的直接对照研究。** 最接近的三条是：①Lc0 的单次前向中已存在前瞻表征（https://arxiv.org/abs/2406.00877）；②Chessformer 用架构/表示换算力而非循环（https://arxiv.org/abs/2605.19091）；③Ruoss 只对比了外部搜索与容量，未做循环前向（https://arxiv.org/abs/2402.04494）。
- 因此，"固定参数量下，把计算换成循环步数（ponder）能否在象棋上换来棋力"目前是一个**公开空白**，也正是我们最值得抢先验证的点（本机 3060 可做小规模对照）。

---

## 八、地形图总结

**五个问题域各自的一句话答案**

1. 免搜索/弱搜索 RL 的证据边界：在不完美信息游戏（Suphx）与象棋类（Xiangqi，天梯前 0.1%）上，无搜索 RL 已能到强人类水平，但稳定化几乎全靠三件事——**对手池、密集/截断的价值目标、受控探索**；纯终局奖励 + 全轨迹 bootstrap 会训崩（Xiangqi 的 GAE 崩溃案例），而象棋上"无搜索 RL 微调"目前只被证明能改战术指标、不自动涨棋力（2608.27757）。
2. 内化搜索的机制设计空间：机制很多（循环/looped、latent CoT、pause token、自适应深度、学习动力学、搜索蒸馏），理论说循环表达力足够模拟多步推理，经验说循环能换参数效率；但**在象棋上没有任何一篇做过"容量 vs 计算步数"的受控对照**，Ruoss 只证明了容量可扩展、搜索的收益未被单次前向吸收。
3. 象棋专属表示与目标设计：**位置表示是否进入注意力（几何偏置）与策略头的 from-to 结构**比堆参数更划算（8–30× 计算节省、+100 Elo in Lc0）；辅助目标（对手回应、ownership/score 类子目标）在 Go 上被消融证明有效，象棋上仍是空白；平面 bitboard 输入有卷积归纳偏置优势，token 序列需要位置偏置补偿。
4. 蒸馏规模律：局面数量在千万级、标注形状（action-value 全走法 > 单标量 > 只有 best move）比数据量更关键；state-value 路线所需规模小 1–2 个数量级（Gigafish 固定深度 10 的 10 亿样本仍欠拟合）；且 loss/准确率不能预测棋力。
5. 评估方法论：用 pentanomial + GSPRT 的社区标准（fishtest/fastchess）做单点比较、用 BayesElo/Ordo 做多档 ladder、永远带锚点与局数/CI、**绝不混用 vs-bots 与 vs-humans 的 Elo**（同一模型两者可差 600 Elo）；puzzle 准确率必须与 ladder Elo 同时报告（两者已被证明解耦）。

**我们项目最该先验证的 3 个假设**

- **H1（目标函数）：把 SL 目标换成"每个合法走法的胜率分布"（K=128 桶分类 + HL-Gauss）比现有 policy 交叉熵 + 标量 value 更利于棋力。**
  验证：在 3060 probe（5–20M 局面子采样）上跑三臂（现方案 / action-value 头 / value-only + 一步 argmax），固定评估集报 puzzle 准确率，并用 fastchess 对 SF17.1 d6 做 ≥200 局 ladder。判据：action-value 臂在 ladder 上不劣且 puzzle 明显更好，则把主目标切过去；否则保持并记录（H1 证伪本身就是一个可写进笔记的结论）。
- **H2（RL 稳定化）：v1 的 RL 退化来自 value/advantage 估计，而非"免搜索 RL 不 work"。**
  验证：从 SL checkpoint 出发做 2×2 消融 {纯自对弈, 动态对手池} × {全轨迹 GAE(γ=λ=1), L 步截断 GAE}，每档跑固定墙钟时间后测对 SF d6 的 ladder + 记录 policy 熵/线路多样性。判据：截断 GAE 或对手池任一条件下 ladder 不退化（≥ 基线），即支持 H2；两条件都崩溃才说明问题在别处（如奖励或数据）。
- **H3（计算步数）：固定参数量下，用 K 步循环/ponder 换计算，比同 FLOPs 的稠密加深更省参数、且收益集中在中等难度局面。**
  验证：在同一 SL 数据与预算下训练 {100M 稠密} vs {循环块 K=2/4 的同参数模型}，比较 puzzle（按难度分档）与 ladder；先用固定 K，确认收益后再加停机门。判据：循环模型在中等难度档的 puzzle 增益 ≥ 稠密模型且参数量相同，即支持 H3；若循环在 K>1 无增益，说明额外计算未被利用，须先解决"如何让模型用上循环步数"（随机深度采样、逐层监督等）。

---

## 引用链接（全部）

**无搜索/弱搜索 RL 与自对弈稳定化**：https://arxiv.org/abs/1712.01815 AlphaZero (2017)｜https://arxiv.org/abs/2402.04494 Ruoss et al. Grandmaster-Level Chess Without Search (2024)｜https://arxiv.org/abs/2410.04865 Xiangqi Without Search (2024)｜https://arxiv.org/abs/2003.13590 Suphx (2020)｜https://arxiv.org/abs/2609.09094 Approximate Value Iteration in Self-Play (2026)｜https://arxiv.org/abs/2608.27757 Prior-Directed Exploration for Searchless Chess (2026)｜https://openreview.net/forum?id=bERaNdoegnO Gumbel AlphaZero (2022)｜https://arxiv.org/abs/2310.11305 MiniZero (2024)

**联盟/对手池/价值目标**：https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/ AlphaStar (2019)｜https://arxiv.org/abs/1711.00832 PSRO (2017)｜https://arxiv.org/abs/1603.01121 Neural Fictitious Self-Play (2016)｜https://ir.cwi.nl/pub/30870/30870.pdf Value targets in off-policy AlphaZero (2021)｜https://arxiv.org/abs/2403.03950 Stop Regressing (2024)

**内化搜索/前瞻**：https://arxiv.org/abs/2412.12119 External & Internal Planning with LMs (ICML 2025)｜https://arxiv.org/abs/2406.00877 Learned look-ahead in Lc0 (2024)｜https://arxiv.org/abs/2505.21552 同上现象的上下文依赖 (2025)｜https://arxiv.org/abs/1705.08439 ExIt (2017)

**latent CoT / 额外 token / 自适应计算**：https://arxiv.org/abs/2412.06769 Coconut (2024)｜https://arxiv.org/abs/2310.02226 Pause tokens (2023)｜https://arxiv.org/abs/2404.15758 Let's Think Dot by Dot (2024)｜https://arxiv.org/abs/2403.09629 Quiet-STaR (2024)｜https://arxiv.org/abs/1603.08983 ACT (2016)｜https://arxiv.org/abs/2107.05407 PonderNet (2021)｜https://arxiv.org/abs/1807.03819 Universal Transformers (2018)｜https://arxiv.org/abs/2207.07061 CALM (2022)

**循环/自适应深度**：https://arxiv.org/abs/2404.02258 Mixture-of-Depths (2024)｜https://arxiv.org/abs/2507.10524 Mixture-of-Recursions (2025)｜https://arxiv.org/abs/2502.05171 Huginn (2025)｜https://arxiv.org/abs/2510.25741 Ouro (2025)｜https://arxiv.org/abs/2502.17416 Looped Transformers / Latent Thoughts (ICLR 2025)｜https://arxiv.org/abs/2301.13196 Looped Transformers as Programmable Computers (2023)｜https://arxiv.org/abs/2506.21734 HRM (2025)｜https://arxiv.org/abs/2510.04871 TRM (2025)｜https://arxiv.org/abs/2510.00355 HRM 批评性补充材料 (2025)｜https://arxiv.org/abs/2606.18208 Looped World Models (2026)｜https://arxiv.org/abs/2602.08864 循环 transformer 的动态计算分配 (2026)

**学习动力学/测试时计算**：https://arxiv.org/abs/1911.08265 MuZero (2019)｜https://arxiv.org/abs/2111.00210 EfficientZero (2021)｜https://arxiv.org/abs/2406.10667 UniZero (2024)｜https://arxiv.org/abs/2408.03314 Scaling test-time compute (2024)

**象棋表示/目标/规模**：https://arxiv.org/abs/2409.12272 Chessformer (2024)｜https://arxiv.org/abs/2605.19091 Chessformer: unified architecture (2026)｜https://blog.lukesalamone.com/posts/distilling-stockfish/ Gigafish / 1B positions (2026)｜https://arxiv.org/abs/1902.10565 KataGo (2019/2020) + https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md｜https://arxiv.org/abs/2510.20783 OOD compositionality in chess transformers (2025)｜https://arxiv.org/abs/2403.15498 Karvonen, chess LM world models (2024)｜https://arxiv.org/abs/2210.13382 Othello-GPT (2023)

**人类建模与规模律**：https://arxiv.org/abs/2006.01855 Maia (2020)｜https://arxiv.org/abs/2409.20553 Maia-2 (2024)｜https://arxiv.org/abs/2008.10086 Learning Models of Individual Behavior in Chess (2020)｜https://arxiv.org/abs/2104.03113 Scaling Scaling Laws with Board Games (2021)｜https://arxiv.org/abs/2210.00849 Scaling laws for a MARL model (2022)｜https://arxiv.org/abs/2412.11979 AlphaZero neural scaling and Zipf's law (2024)｜https://lczero.org/dev/wiki/technical-explanation-of-leela-chess-zero/ Leela Chess Zero 技术文档

**评估方法论**：https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html Fishtest 统计方法｜https://www.chessprogramming.org/Sequential_Probability_Ratio_Test SPRT 实践与常用边界｜https://github.com/jw1912/SPRT SPRT/GSPRT 推导与脚本｜https://github.com/Disservin/fastchess/blob/master/man.md fastchess 手册｜https://www.chessprogramming.org/Engine_Testing 引擎测试与开局库｜https://inria.hal.science/inria-00323349/document Whole-History Rating（Coulom, 2008）｜https://ijccrl.com/bayeselo-and-ordo-in-computer-chess/ BayesElo / Ordo 在引擎评分中的使用｜https://beuke.org/chess-engine-draws/ 引擎和棋率饱和｜https://arxiv.org/pdf/2105.00839 联赛 Elo 的不确定度｜https://arxiv.org/abs/2410.03893 Allie: Human-aligned Chess with a Bit of Search (2024)

**2024–2026 LLM/棋类新工作**：https://arxiv.org/abs/2507.00726 RLVR in chess (2025)｜https://arxiv.org/abs/2507.12215 Xiangqi-R1 (2025)｜https://arxiv.org/abs/2509.24239 ChessArena (2025)｜https://arxiv.org/abs/2512.01992 LLM Chess (2025)｜https://arxiv.org/abs/2603.20510 Grounded Chess Reasoning via Master Distillation (2026)｜https://arxiv.org/abs/2604.05134 How Reasoning Evolves from Post-Training Data: Chess (2026)｜https://arxiv.org/abs/2606.25176 Engine-Agnostic Search with Human Policy Guidance (2026，索引项)
