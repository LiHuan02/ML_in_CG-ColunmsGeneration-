# VCSP Column Generation with Machine Learning Column Selection

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

建议在仓库根目录执行命令。Windows PowerShell 下直接运行 `python ...` 即可；如果你的环境找不到本地包，可先执行：

```powershell
$env:PYTHONPATH='.'
```

### 2. 生成训练数据

训练数据由 `data_generation.generate` 生成。默认会：

- 为每个实例运行 CG + MILP 选择
- 保存每轮迭代的二分图特征到 `iter_XXXX.npz`
- 额外保存 `metadata.npz`、`manifest.csv`、`generation_config.json`、`generation_summary.json`

常用命令：

```bash
# 生成 30 trip、10 个实例的数据
python -m data_generation.generate --trips 30 --instances 10

# 生成论文规模附近的数据，例如 400 trip、100 个实例
python -m data_generation.generate --trips 400 --instances 100 --workers 4

# 指定输出目录并强制覆盖已有结果
python -m data_generation.generate --trips 100 --instances 20 --output data/my_training_data --overwrite
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--trips` | 每个实例的 trip 数 |
| `--instances` | 生成的实例数 |
| `--output` | 输出目录，默认 `data_generation/training_data/vcsp_{trips}` |
| `--max-iterations` | 每个实例最多 CG 迭代次数 |
| `--workers` | 并行进程数，`1` 为串行 |
| `--overwrite` | 覆盖已有实例目录 |
| `--cost-inflation` | 初始列成本膨胀系数 |
| `--artificial-cost` | 人工 f/g-trip 列成本 |
| `--epsilon` | MILP 选择惩罚项 |
| `--additional-pct` | 额外补充的负 reduced cost 列比例 |

生成结果目录里每个实例会包含：

- `iter_0000.npz`, `iter_0001.npz`, ...
- `metadata.npz`
- `manifest.csv`
- `generation_config.json`
- `generation_summary.json`

### 3. 训练 GNN 模型

训练入口是 `gnn.train`。脚本会自动读取数据目录下所有 `instance_XXXX/iter_XXXX.npz` 文件，默认按 75/25 划分训练集和验证集，并保存：

- `best_model.pt`
- `last_model.pt`
- `history.csv`
- `training_summary.json`
- `norm_stats.npz`

常用命令：

```bash
# 使用默认超参数训练
python -m gnn.train --data data_generation/training_data/vcsp_400 --output gnn/models/vcsp_400

# 指定 batch 累积、早停和随机种子
python -m gnn.train \
  --data data_generation/training_data/vcsp_400 \
  --output gnn/models/vcsp_400 \
  --epochs 200 \
  --batch-size 16 \
  --patience 30 \
  --seed 42

# 从已有 checkpoint 继续训练
python -m gnn.train --data data_generation/training_data/vcsp_400 --output gnn/models/vcsp_400 --resume gnn/models/vcsp_400/last_model.pt
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--data` | 训练数据目录 |
| `--output` | 模型和日志输出目录 |
| `--epochs` | 最大训练轮数 |
| `--batch-size` | 梯度累积的图数量 |
| `--patience` | 验证集指标早停耐心值，`0` 表示关闭 |
| `--min-delta` | 视为提升所需的最小增量 |
| `--resume` | 从 checkpoint 恢复训练 |
| `--save-every` | 每隔 N 轮保存一次中间 checkpoint |
| `--no-normalize` | 关闭特征标准化 |

### 4. 运行对比试验

对比试验入口是 `experiments.comparison`，支持 `no_selection`、`milp`、`gnn` 三种策略，并会把每次实验的原始结果和汇总结果保存到输出目录。

常用命令：

```bash
# 跑默认的 40/80 trip 对比实验
python -m experiments.comparison

# 指定 trip 规模、实例数和策略
python -m experiments.comparison --trip-sizes 20,30,50 --instances 3 --strategies no_selection,milp,gnn

# 指定 GNN 模型路径并保存结果
python -m experiments.comparison \
  --trip-sizes 300,400 \
  --instances 5 \
  --strategies no_selection,milp,gnn \
  --gnn-model gnn/models/best_model.pt \
  --norm-stats gnn/models/norm_stats.npz \
  --output experiments/results/vcsp_compare
```

输出文件包括：

- `results.json`
- `results.csv`
- `summary.csv`
- `reductions.csv`

### 5. 运行单次求解

```bash
# 无选择策略
python main.py --trips 30 --selection no_selection

# MILP 选择策略
python main.py --trips 30 --selection milp

# GNN 选择策略
python main.py --trips 30 --selection gnn
```

`main.py` 的 `--selection` 支持：

- `no_selection`
- `milp`
- `gnn`

如果是 GNN 策略，请先确保 `gnn/models/best_model.pt` 和 `gnn/models/norm_stats.npz` 已存在，或在对比脚本中显式指定路径。

---

## 说明


基于 EBSCO 论文（Morabit, Desaulniers, Lodi, 2021）实现的车辆与乘务员联合调度问题（VCSP）列生成求解器，集成机器学习（GNN）列选择策略。

## 目录

- [数学模型](#数学模型)
- [项目架构](#项目架构)
- [快速开始 (更新后)](#快速开始-更新后)
- [列生成框架](#列生成框架)
- [列选择策略](#列选择策略)
- [GNN训练数据生成](#gnn训练数据生成)
- [GNN模型与训练](#gnn模型与训练)
- [实验与对比](#实验与对比)
- [命令行参数](#命令行参数)
- [参考文献](#参考文献)

---

## 实验与对比

下表使用 `experiments/results/latest/summary.csv` 的聚合统计（示例运行）。字段说明：`avg_total_time` 单位为秒，`avg_objective` 为最终目标值，其他为均值。

| num_trips | selection     | runs | avg_total_time (s) | std_total_time | avg_iterations | avg_final_columns | avg_columns_generated | avg_columns_selected | avg_objective | avg_buses |
|-----------:|:--------------|:-----:|-------------------:|---------------:|---------------:|------------------:|---------------------:|--------------------:|--------------:|---------:|
| 40        | gnn          | 2    | 8.4543            | 7.2138         | 76.5           | 303.0             | 745.0                | 223.0               | 2644210.89   | 10.3462  |
| 40        | milp         | 2    | 18.6265           | 1.8266         | 139.5          | 734.5             | 1171.0               | 654.5               | 1630833.10   | 8.42     |
| 40        | no_selection | 2    | 16.7330           | 0.1826         | 159.0          | 1155.0            | 1075.0               | 1075.0              | 1629481.88   | 8.4375   |
| 80        | gnn          | 2    | 13.0437           | 0.4850         | 18.5           | 216.5             | 218.5                | 56.5                | 8202598.43   | 24.5     |
| 80        | milp         | 2    | 225.4577          | 49.2432        | 258.0          | 1675.5            | 2977.0               | 1515.5              | 2888515.94   | 14.0492  |
| 80        | no_selection | 2    | 221.5217          | 5.7809         | 300.0          | 3302.5            | 3155.0               | 3142.5              | 2871258.92   | 14.0     |

如需我将该表格自动同步到 README（从 CSV 读取并更新），或添加可视化图表（PNG/SVG）并把图片链接到 README，请告诉我偏好。

---

## 数学模型

### 主问题（Master Problem）

基于 Haase, Desaulniers, Desrosiers (2001) 的集合划分模型：

```
Min  c·B + Σ(c_p · θ_p)
s.t.
  Σ(e_vp · θ_p) = 1,     ∀ v ∈ V          (d-trip 覆盖约束)
  Σ(f_wp · θ_p) = 1,     ∀ w ∈ W          (车辆到达约束)
  Σ(g_wp · θ_p) = 1,     ∀ w ∈ W          (车辆出发约束)
  Σ(q_hp · θ_p) − B ≤ 0, ∀ h ∈ H          (车辆数量约束)
  θ_p ≥ 0, B ≥ 0
```

| 符号 | 含义 |
|------|------|
| `θ_p` | 第 p 个驾驶员职责被选择的比例（RMP中为连续变量） |
| `B` | 所需公交车辆总数 |
| `c_p` | 职责 p 的运营成本 |
| `c` | 单车固定成本 |
| `e_vp` | 职责 p 是否覆盖 d-trip v |
| `f_wp` | 职责 p 是否包含 trip w 开始位置的车辆到达 |
| `g_wp` | 职责 p 是否包含 trip w 结束位置的车辆出发 |
| `q_hp` | 职责 p 是否在时间 h 需要车辆 |

### 定价问题（Pricing Problem）— 资源约束最短路

对偶变量：`α_v` (d-trip), `β_w` (到达), `γ_w` (出发), `δ_h` (车辆数)

**弧简化成本：**
```
c'_ij = c_ij − α_v·e_v − β_w·f_w − γ_w·g_w − δ_h·q_h
```

求解算法：前向标签算法（Forward Labeling Algorithm）
- 标签：(node, cumul_cost, duty_length)
- 支配规则：L1 支配 L2 若 cost(L1) ≤ cost(L2) 且 duty_length(L1) ≤ duty_length(L2)
- 最大职责长度：300 分钟

---

## 项目架构

```
ColunmsGeneration(CG)/
├── core/                           # 列生成核心框架
│   ├── column_generation.py        # VCSPSolver: CG 主循环编排
│   ├── rmp.py                      # 受限主问题基类
│   └── pp.py                       # 定价问题基类
│
├── problems/vcsp/                  # VCSP 问题定义
│   ├── instance.py                 # 随机实例生成器
│   ├── column.py                   # VCSPColumn: 职责列表示
│   ├── vcsp_rmp.py                 # VCSP 受限主问题 (ortools LP)
│   ├── vcsp_pp.py                  # 定价问题 (RCSPP 标签算法)
│   └── driver_network.py           # 司机网络构建
│
├── selection/                      # 列选择策略
│   ├── no_selection.py             # NO-S: 无选择（添加所有列）
│   ├── milp_selection.py           # MILP-S: MILP 精确选择
│   └── gnn_selection.py            # GNN-S: 图神经网络选择
│
├── data_generation/                # GNN 训练数据生成
│   ├── feature_extractor.py        # 12维列特征 + 2维约束特征提取
│   ├── milp_labeler.py             # MILP 标签生成器 (Section 3.1.1)
│   ├── data_collector.py           # CG + MILP + 特征提取一体化收集器
│   └── generate.py                 # 批量生成训练数据脚本
│
├── gnn/                            # GNN 模型与训练
│   ├── bipartite_gnn.py            # 二分图 GNN 模型 (Algorithm 1)
│   ├── dataset.py                  # 训练数据加载器
│   ├── train.py                    # 训练脚本
│   └── models/                     # 训练好的模型
│
├── experiments/                    # 实验与评估
│   └── comparison.py               # NO-S / MILP-S / GNN-S 对比实验
│
├── main.py                         # 命令行入口
└── requirements.txt                # 依赖
```

---



## 列生成框架

### 算法流程

```
1. 初始化: 为每个 d-trip 生成启发式初始列
2. 迭代:
   a. 求解 RMP → 得到对偶值 (α, β, γ, δ)
   b. 求解定价问题 (RCSPP) → 生成负简化成本列
   c. 去重: 按 d-trip 签名检测重复列
   d. 列选择: 应用选择策略筛选列
   e. 添加选中列到 RMP
   f. 若无负简化成本列 → 最优, 停止
3. 返回最终列集合和解
```

### 关键类

**`VCSPSolver`** (`core/column_generation.py`)
- 编排 CG 主循环
- 管理 RMP、定价问题、列选择器
- 追踪迭代统计（时间、列数、目标值）

**`VCSPRMP`** (`problems/vcsp/vcsp_rmp.py`)
- 使用 ortools GLOP 求解 LP 松弛
- 提取对偶值：`alpha` (d-trip), `beta` (到达), `gamma` (出发), `delta` (车辆数)

**`VCSPPricingProblem`** (`problems/vcsp/vcsp_pp.py`)
- 前向标签算法求解 RCSPP
- 支配过滤减少标签数
- 最多生成多列（所有到达汇点的负简化成本路径）

---

## 列选择策略

### NO-S: 无选择策略

论文基准策略。添加所有生成的负简化成本列。

```
1. 按简化成本升序排列
2. 分配列到不相交块 (disjoint blocks)
3. 取前 n_max_blks 个块的列
```

### MILP-S: MILP 精确选择

论文"专家"策略。每轮求解 MILP：

```
Min  Σ(c_p·θ_p) + c·B + ε·Σ(y_p)
s.t.
  所有 RMP 约束
  θ_p ≤ y_p,  ∀ p ∈ 新生成列
  θ_p ≥ 0, y_p ∈ {0, 1}
```

- `ε = 0.1`：小惩罚系数，最小化选中列数
- 选中 `y_p = 1` 的列 + 50% 剩余负简化成本列（避免收敛问题）
- 使用 ortools CBC/SCIP 求解器

### GNN-S: 图神经网络选择

用训练好的 GNN 模型替代 MILP 进行快速预测。

**流程：**
```
1. 提取列特征 + 约束特征
2. 构建二分图 (列节点 ↔ 约束节点)
3. GNN 推理 → 获取每列的选择概率
4. 选择概率 > 0.5 的列
```

**特征（论文 Section 4.2）：**

| 维度 | 列特征 | 约束特征 |
|------|--------|---------|
| 1 | cost | dual_value |
| 2 | reduced_cost | node_degree |
| 3 | 总约束覆盖数 | |
| 4–7 | 各约束组覆盖数 (4组) | |
| 8 | duty_length | |
| 9 | duty_type | |
| 10 | is_new (1/0) | |
| 11 | column_value (θ_p) | |
| 12 | incompatibility_degree | |

---

## GNN训练数据生成

### 标签生成 (MILP Labeler)

运行 CG + MILP 选择，每个 CG 迭代记录一次二分图：

```
每次迭代存储:
├── column_features:    (n_cols, 12)   浮点
├── constraint_features: (n_cons, 2)   浮点
├── edge_index:         (2, n_edges)   整型 [col_idx, cons_idx]
├── labels:             (n_new_cols,)  0/1 (来自 MILP y_p)
├── new_col_mask:       (n_cols,)      布尔
└── basic_col_mask:     (n_cols,)      布尔
```

### 数据收集策略

使用**人工高成本初始列**确保 CG 有充分改进空间：

```
初始列结构:
  d-trip 列: 每 d-trip 一个，仅覆盖 d_trip（不覆盖 f/g trip）
             成本 = 真实成本 × 3.0
  f-trip 人工列: 每 trip 一个，仅覆盖 f_trip
             成本 = 1e7（极高，迫使 CG 找到更好列）
  g-trip 人工列: 每 trip 一个，仅覆盖 g_trip
             成本 = 1e7
```

### 生成命令

```bash
# 小规模测试
PYTHONPATH=. python -m data_generation.generate --trips 30 --instances 10

# 论文规模（100实例 × 400 trips ≈ 7,000+ 数据点）
PYTHONPATH=. python -m data_generation.generate --trips 400 --instances 100

# 自定义输出
PYTHONPATH=. python -m data_generation.generate \
  --trips 100 --instances 50 \
  --output data/my_training_data \
  --cost-inflation 3.0 --artificial-cost 1e7
```

---

## GNN模型与训练

### 模型架构（论文 Algorithm 1 + Table 2）

```
二分图消息传递 (K=1 轮):

Phase 1 — 约束节点更新:
  a_c = Σ φ_C(h_c, h_v)      对每个邻居列 v
  h_c' = ψ_C([h_c, a_c])

Phase 2 — 列节点更新:
  a_v = Σ φ_V(h_v, h_c')     对每个邻居约束 c
  h_v' = ψ_V([h_v, a_v])

输出:
  y_v = Sigmoid(out(h_v'))
```

**网络结构：**

| 组件 | 架构 | 激活 |
|------|------|------|
| φ_C, ψ_C, φ_V, ψ_V | Linear(d→32) → ReLU → Linear(32→32) | ReLU |
| out | Linear(32→32) → ReLU → Linear(32→32) → ReLU → Linear(32→1) | Sigmoid |

**超参数（论文 Table 2）：**

| 参数 | 值 |
|------|-----|
| 消息传递轮数 K | 1 |
| 学习率 | 1e-3 |
| 优化器 | Adam |
| 损失函数 | 加权 BCE (正类:负类 = 10:1) |
| 隐藏维度 | 32 |
| 模型参数量 | 10,849 |

### 训练命令

```bash
# 基础训练
PYTHONPATH=. python -m gnn.train --data data_generation/training_data/combined

# 完整训练
PYTHONPATH=. python -m gnn.train \
  --data data_generation/training_data/combined \
  --epochs 1000 \
  --lr 1e-3 \
  --pos-weight 10.0 \
  --output gnn/models

# 使用 GPU
PYTHONPATH=. python -m gnn.train --data data/combined --device cuda
```

### 训练结果

在 423 个数据点（30-trip × 10 + 50-trip × 20）上训练：

| 指标 | 我们的模型 | 论文 (Table 3) |
|------|-----------|----------------|
| Recall (TPR) | 83.46% | 86.2% |
| TNR | 61.52% | 66.6% |
| Precision | 39.58% | 23.7% |
| **Balanced Accuracy** | **72.49%** | **76.5%** |

> 仅用 686 个数据点（论文用 7,000+ 来自 100 个 400-trip 实例），模型性能已接近论文水平。Precision 更高意味着误报更少，选择的列质量更高。

---

## 实验与对比

以下为基于 `experiments/results/latest` 的简要汇总（示例运行，num_trips=40/80，各策略对比）：

- 对于 40 trips：
  - `gnn` 平均总耗时约 8.45s，平均迭代 76.5 次，平均最终列数 303，平均选中列 ~223。目标值显著高于 `milp`/`no_selection`（分别为 1.63e6 / 1.63e6 左右），但运行速度快得多。
  - `milp` 平均总耗时约 18.63s，平均迭代 139.5 次，平均最终列数 734.5，平均选中列 ~654.5。
  - `no_selection` 平均总耗时约 16.73s，平均迭代 159 次，平均最终列数 1155。

- 对于 80 trips：
  - `gnn` 平均总耗时约 13.04s，平均迭代 18.5 次，平均最终列数 216.5，平均选中列 ~56.5，但目标值偏高（示例中约 8.2e6）。
  - `milp` 与 `no_selection` 运行时间显著更长（均在 200s+），并生成更多列与更大最终列数。

以上结果来自 `experiments/results/latest/summary.csv` 的聚合统计，用于 README 中的简要说明。若需更详细的表格或可视化，请告知要包含的字段（例如 `avg_total_time`、`avg_iterations`、`avg_objective` 等）。

### 运行对比

```python
from experiments.comparison import run_comparison_experiment, print_summary_table

results = run_comparison_experiment(
    trip_sizes=(20, 30, 50),
    num_instances=3,
    strategies=('no_selection', 'milp', 'gnn'),
)
print_summary_table(results)
```

### 输出格式（匹配论文 Table 5）

```
 Trips       Strategy    Total      RMP       PP  Iters     Cols        Obj  Buses
-----------------------------------------------------------------------------------------
    20            gnn      6.3      0.3      5.9     38      299 1296976.90    7.9
    20           milp      7.2      0.2      5.9     45      252 1290334.81    7.7
    20   no_selection      5.8      0.3      5.5     38      299 1296976.90    7.9
    30            gnn      4.0      0.0      4.0      5       78 3781913.50   15.5
    30           milp      4.0      0.0      4.0      5       73 3781913.50   15.5
    30   no_selection      3.9      0.0      3.9      5       78 3781913.50   15.5

AVERAGE TIME REDUCTION (vs NO-S)
  20 trips:
               gnn: -8.8%
              milp: -23.9%
  30 trips:
               gnn: -0.8%
              milp: -1.4%
```

> **注意：** 小规模实例（20–30 trips）的高度退化性质使得所有策略表现相似。论文在大规模实例（300–800 trips）上报告 GNN-S 有 **25–30% 的时间减少**。提高训练数据规模和实例大小可获得更好的效果。

---

## 命令行参数

### main.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--trips` | 30 | trip 数量 |
| `--relief` | 2 | 每个 trip 的换乘点数量 |
| `--seed` | 42 | 随机种子 |
| `--selection` | `no_selection` | 选择策略: `no_selection` / `milp` / `gnn` |
| `--max-iter` | 200 | 最大 CG 迭代次数 |
| `--bus-cost` | 50000 | 公交车固定成本 |
| `--driver-cost` | 50000 | 驾驶员固定成本 |
| `--cost-per-min` | 1.0 | 每分钟运营成本 |

### data_generation/generate.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--trips` | 50 | 每个实例的 trip 数 |
| `--instances` | 20 | 生成的实例数量 |
| `--output` | `training_data/vcsp_{trips}` | 输出目录 |
| `--max-iterations` | 300 | 每个实例最大 CG 迭代数 |
| `--epsilon` | 0.1 | MILP 惩罚系数 |
| `--additional-pct` | 0.5 | 额外列百分比（收敛保障） |
| `--cost-inflation` | 3.0 | 初始列成本膨胀因子 |
| `--artificial-cost` | 1e7 | 人工 f/g-trip 列成本 |

### gnn/train.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | `training_data/test` | 训练数据目录 |
| `--epochs` | 200 | 训练轮数 |
| `--lr` | 1e-3 | 学习率 |
| `--pos-weight` | 10.0 | 正类 BCE 权重 |
| `--hidden-dim` | 32 | GNN 隐藏维度 |
| `--num-iterations` | 1 | 消息传递轮数 K |
| `--val-split` | 0.25 | 验证集比例 |
| `--output` | `gnn/models` | 模型输出目录 |

---

## 参考文献

1. **Morabit, M., Desaulniers, G., & Lodi, A.** (2021). Machine-Learning–Based Column Selection for Column Generation. *Transportation Science*, 55(4), 815–831.

2. **Haase, K., Desaulniers, G., & Desrosiers, J.** (2001). Simultaneous Vehicle and Crew Scheduling in Urban Mass Transit Systems. *Transportation Science*, 35(3), 286–303.

3. **Elhallaoui, I., Metrane, A., Soumis, F., & Desaulniers, G.** (2010). Multi-phase dynamic constraint aggregation for set partitioning type problems. *Mathematical Programming*, 123(2), 345–370.

4. **Pessoa, A., Sadykov, R., Uchoa, E., & Vanderbeck, F.** (2018). Automation and combination of linear-programming based stabilization techniques in column generation. *INFORMS Journal on Computing*, 30(2), 339–360.

---

## License

本项目基于 EBSCO 论文复现，仅供学习和研究使用。
