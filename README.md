# Lightning OPD Toy Experiment

**题目二:Offline OPD under Agentic Shift —— 离线 OPD 的假设边界**

复现 Lightning OPD(arXiv:2604.13010)的核心思想,在 multi-turn agentic toy task 上对比五种
方法,实证 offline OPD 的成立条件与失效边界,并提出一个 DAgger-style patch。

> 完整答题见 **`答题_题目二.md`**。

## 核心结论

- **强参考 + 充分覆盖** → offline OPD = online OPD = 100%(论文 modest-drift 假设成立)。
- **覆盖不足** → offline OPD 因 distribution shift 崩溃(31.6%,off-support 0.66),
  online OPD 仍 96.8%。
- **DAgger refresh patch** → off-support 0.66→0.20,成功率恢复到 99.8%。

## Toy Task:Multi-hop KG-QA

```
图谱:  Person --[works_at]--> Company --[located_in]--> Country
任务:  给定 person P,找到其所在 country
最优:  lookup_person(P) → lookup_company(C) → answer(N)   [3 步]
```

满足题目五条件:多轮决策、工具式动作、错误分支、最终答案稀疏奖励、早期错误影响后续状态。
**关键设计**:图谱每 episode 随机重置,边只能靠 lookup 揭示——不查就答不出,真正强制多跳推理
与错误传播(详见 `答题_题目二.md` (iii))。

## 文件结构

```
.
├── env.py                # KGQAEnv:每 episode 随机重置的多跳 KG 环境
├── model.py              # PolicyNet:MLP 策略网络
├── collect.py            # 轨迹收集 + OfflineDataset(忠实 replay 预计算 teacher label)
├── train.py              # 五种训练算法
├── evaluate.py           # 评估工具
├── plot.py               # 结果可视化
├── main.py               # 主实验脚本(含可复现的 per-method 种子)
├── run_experiment.sh     # 快速运行脚本
├── 答题_题目二.md         # ⭐ 题目二完整回答(i)-(vii)
└── results/ results_shift/  # 两套机制的结果(json + 3 张图)
```

## 安装与运行

```bash
pip install numpy torch matplotlib

# 机制 A(强参考,modest drift 成立):
python main.py --plot --output_dir ./results

# 机制 B(部分覆盖,distribution shift 失效 + DAgger 修复):
python main.py --plot --n_offline 150 --sft_epochs 40 --opd_epochs 30 \
    --rl_episodes 2000 --n_eval 500 --output_dir ./results_shift
```

## 五种方法

| 方法 | rollout 来源 | 训练信号 | 实时 teacher |
|---|---|---|---|
| SFT | oracle 轨迹 | CE 模仿 oracle | 否 |
| Online RL | student `π_θ` | 任务回报 advantage | 否 |
| Offline OPD | 固定 `π_ref` | `A_t=log π_T−log π_θ`,预计算 | 否 |
| Online OPD(上界) | student `π_θ` | `A_t=log π_T−log π_θ`,实时 | 是 |
| DAgger OPD(patch) | offline + 周期 student refresh | teacher-action 行为克隆 | 周期性 |

OPD 的 advantage 定义为 `A_t = log π_T(a_t|s_t) − log π_θ(a_t|s_t)`(stop-grad,clip),最大化
等价于最小化 reverse KL `KL(π_θ‖π_T)`。online/offline 只差 rollout 分布。

## 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--n_offline` | 300 | 离线 oracle 轨迹数 |
| `--sft_epochs` | 60 | SFT 训练轮数 |
| `--opd_epochs` | 40 | OPD 训练轮数 |
| `--rl_episodes` | 3000 | RL/online OPD episode 数 |
| `--dagger_refresh_every` | 8 | DAgger 刷新间隔(epoch) |
| `--n_eval` | 500 | 最终评估 episode 数 |
| `--seed` | 42 | 随机种子(per-method 固定偏移,可复现) |

## 论文对应

- Lightning OPD 核心:`offline_opd_train()`(预计算 `log π_T` 复用)
- online 上界:`online_opd_train()`(实时 oracle 打分)
- distribution shift 追踪:`_off_support_ratio()`
- DAgger patch:`dagger_opd_train()`
- 对比论文 2509.26497(forward-KL 软标签蒸馏)的异同见 `答题_题目二.md` (vi)

</content>
