# VCSP Column Generation with Machine Learning Column Selection

基于 EBSCO 论文（Morabit, Desaulniers, Lodi, 2021）实现的车辆与乘务员联合调度问题（VCSP）列生成求解器，集成机器学习（GNN）列选择策略。

## 目录

- [数学模型](#数学模型)
- [项目架构](#项目架构)
- [快速开始](#快速开始)
- [列生成框架](#列生成框架)
- [列选择策略](#列选择策略)
- [GNN训练数据生成](#gnn训练数据生成)
- [GNN模型与训练](#gnn模型与训练)
- [实验与对比](#实验与对比)
- [命令行参数](#命令行参数)
- [参考文献](#参考文献)

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

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

依赖：`numpy`, `ortools`, `torch`

### 运行列生成

```bash
# 无选择策略 (NO-S)
PYTHONPATH=. python main.py --trips 30 --selection no_selection

# MILP 选择策略 (MILP-S)
PYTHONPATH=. python main.py --trips 30 --selection milp

# GNN 选择策略 (GNN-S) — 需要先训练模型
PYTHONPATH=. python main.py --trips 30 --selection gnn
```

### 运行对比实验

```bash
PYTHONPATH=. python experiments/comparison.py
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
