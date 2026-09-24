# v0.9.1 release validation (教材 PDF)

## 教材构建与两轮视觉验收

- 构建管线: `scripts/md2tex_book.py` (Markdown → ctexbook, xelatex;
  系统 TeX Live 2025, 零额外 Python 依赖); xelatex 三遍 (目录/书签)。
- 终版 PDF: 107 页 — 封面 + 210 条可点击目录 + 页眉页脚页码 + 12 张
  真实计算配图; 0 LaTeX 错误、0 缺字形、0 纵向溢出、0 空白页, 正文/
  表格/图形无一边界越界 (CJK 标点悬挂 ≤8pt 属正常排版)。
- 验收方式: 独立审阅代理以 PyMuPDF 逐 span 几何测量 (页 595.28×841.89pt,
  文本块 x 68–527.3, 两轮)。
- 两轮验收修复的阻断项: 页眉页脚 `\small` 被当文字打印 (82 页无页码);
  封面 `\fontsize` 花括号被 f-string 吞掉 (标题 4pt); 代码块 BookCode
  环境丢失 (长路径越界 58pt、`--` 变 en dash); 表格被 `|psi|` 误分列
  且 85 行术语表不可断页 (38 行丢失、1.2pt 不可读字号); 附录表字面
  LaTeX; 段首缩进推移表格 17.9pt; 行内代码 API 名不可断行。
- 保留的正常排版现象: 5–6pt 下标 (表格内), CJK 标点悬挂 ≤8pt。
- 附录数据行核对: 附录 A 129 行 + 附录 B 29 行与 Markdown 源一一对应;
  12 图零畸变、图注齐全。
- 代码测试 56 项通过 (与 v0.9.0 相同)。

---

# v0.9.0 release validation

## Scope

**教材 + 配图版本**: 新增第 6 章《分子振动与能量转移》; 章节重编排
(06-11 → 07-12); 12 张配图由 `scripts/make_book_figures.py` 基于
真实计算生成。动力学与 NN 数值结论不变。

## 配图的独立审阅 (本版的核心工作)

全部 12 张图经独立审阅代理逐张核对源码与物理, 修复 6 处问题:
谐振子本征函数换用物理学家 Hermite (原用 Legendre/概率型, v=1/v=2
不是本征函数); ZPE 占据图温度范围与单调方向修正; LEPS 图改用真
Jacobi 映射 (原键长坐标却标 Jacobi 并画 R=1.5r 分界面, 势垒标注
0.14→实测 ~0.13 au); NN 图标题改为数据驱动 (原标题与自身数据矛盾,
现与回归测试同配置且不复现改善即告警); Morse 能级改用精确非谐公式;
sech² 垒术语与 2D 快照色标/有限时间说明修正。修复后程序化验证:
物理型 Hermite 节点/宇称结构正确 (v=2: 2 节点偶宇称; v=1 奇宇称
零点过原点)。

## 测试

56 项测试通过 (含配图冒烟: 真实渲染、物理质量护栏、配图齐全)。

## 诚实边界 (本版最重要的一条)

**真实量子化学后端在本仓库发布验证环境中未安装、未经端到端验证**:
发布机器上无 `xtb` 可执行文件, 无 pyscf / ase / geometric 包。
因此本版验证的是: (a) 后端协议的实现正确性 (通过 Analytic 与
demo-LJ 后端); (b) 采样管线的几何/滤波/持久化正确性; (c) CLI
闭环端到端可运行。`XTBCommandCalculator` / `PySCFCalculator` /
`ASECalculatorAdapter` 的数值正确性由所调外部程序与参数决定,
本仓库不做认证, 首次使用前应自行做基准校验。

## 已验证项

| 项 | 结果 |
|---|---|
| demo-LJ 解析梯度 vs 中心差分 | rtol 1e-5 内一致 |
| `AnalyticCalculator` FD 回退 vs 解析梯度 | rtol 1e-5 内一致 |
| 采样器: 形状 (n,3N) 特征 + (n,N,3) 几何梯度、ranges 广播、最小距离过滤、max_points 子采样 | 单元测试通过 |
| npz 往返 (points/energies/gradients/provenance/geometry) | 逐数组一致 |
| CLI sample→fit 闭环 (demo 后端, 216 点, 力训练 1500 轮) | 能量 RMSE 9.2e-4 Hartree (0.32% span), 梯度 RMSE 2.4e-3 Hartree/Bohr |
| 缺失后端的报错路径 (`xtb` 未安装) | 抛出含安装指引的 `CommandBackendError` |
| `available_calculators()` | 正确报告 analytic/demo 可用, xtb/pyscf/ase 不可用, 不抛异常 |

## 局限 (已写入 README 与教材 3.6)

- 采样特征为展平笛卡尔坐标: 无置换/旋转等变性 (同核体系需对称性
  扩展, 见第 10 章);
- 采样为均匀位移网格, 非主动学习 (接口位置已预留);
- 后端 provenance (程序/版本/基组/电荷/自旋) 随数据集保存, 是
  可复现性的前提。

## 测试与 wheel 验证

- 全量 53 项测试通过 (macOS arm64, Python 3.14.5), 65 s。
- 全新 venv 安装 `autoquantum-0.8.0-py3-none-any.whl`: 版本元数据
  0.8.0; 53 项测试通过; 隔离环境内 `sample`/`fit`/`backends` CLI
  闭环通过; `pip check` 干净。

Wheel SHA-256:

```text
dd1092aaad1b5109697eba4d3fabb46da13ab57605668fe5f84607b89864fcf7
```

## 遗留

主动学习采样与对称性特征; 四原子以上体系; GPU 加速; 真实后端的
基准验证套件 (待环境具备后补充)。
