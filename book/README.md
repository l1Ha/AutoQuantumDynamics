# 《从势能面到波包》

**分子反应动力学的量子理论、数值验证与代码实践**

从量子力学基础到可验证的分子反应动力学计算——与 [AutoQuantum](../) 代码库逐行对应的中文书籍 (v0.20.0)。

## 适读人群

- **新手**: 有微积分与基础物理背景, 第一次接触量子散射/分子动力学。第 1-3 章是入口, 每章都有"新手路径"专栏。
- **进阶者**: 要读懂或改造本仓库代码的开发者。建议按 4 → 5 → 13 → 8 → 14 → 15 章顺序阅读。
- **高手**: 关心算法选择、数值稳定性与验证严格性的研究者。各章"高手专栏"、第 10/14 章与附录 A/B 面向此层。

## 章节结构

| 部分 | 章节 | 内容 |
|---|---|---|
| 序 | [00 前言与学习路径](chapters/00-前言与学习路径.md) | 本书缘起、三条学习路线、符号与单位约定 |
| I 基础 | [01 数学预备](chapters/01-数学预备.md) | 复数空间、DFT/卷积定理、采样与 Nyquist、差分与求积 |
| | [02 量子力学基础](chapters/02-量子力学基础.md) | Born-Oppenheimer、定态/含时方程、期望值、原子单位 |
| II 势能面 | [03 势能面](chapters/03-势能面.md) | 键坐标 vs Jacobi 坐标、约化质量、Morse/LEPS/Eckart、数据接口 |
| III 电子结构 | [04 电子结构基础](chapters/04-电子结构基础.md) | Born-Oppenheimer 变分求解、基组与高斯基函数、SCF/Roothaan、DFT (LDA/GGA)、半经验 xTB、梯度与力 |
| IV 机器学习 | [05 神经网络势能面](chapters/05-神经网络势能面.md) | 反向传播、Adam、标准化、力训练与双反向传播 |
| | [13 对称函数势能面与主动学习](chapters/13-对称函数势能面与主动学习.md) | Behler–Parrinello 对称函数、共享原子能量委员会、置换不变性、OOD 不确定性与主动学习闭环 |
| V 动力学 | [06 分子振动与能量转移](chapters/06-分子振动与能量转移.md) | 简正振动、Morse 势非谐振、本征态谱、零点能 ZPE |
| | [07 定态散射](chapters/07-定态散射.md) | 匹配条件、透射/反射、Numerov、实验性 2D 求解器 |
| | [08 含时波包](chapters/08-含时波包.md) | 高斯波包、分裂算符、CAP、概率记账恒等式 |
| | [09 二维反应动力学](chapters/09-二维反应动力学.md) | 交换分界面、能量平均透射率、阈值行为 |
| | [14 准经典轨线动力学 (QCT)](chapters/14-准经典轨线动力学.md) | Hamilton 方程、Velocity Verlet 辛积分、Wigner 相空间采样、经典-量子对应与隧穿对拍 |
| | [15 热速率常数与集群生产](chapters/15-热速率常数与集群生产.md) | Boltzmann 态求和、Arrhenius 活化能拟合、Slurm 作业阵列分片并行、micromamba 环境隔离 |
| VI 工程 | [10 数值验证](chapters/10-数值验证.md) | 收敛研究、解析基准、有限差分门控、恒等式测试、仲裁工作流 |
| | [11 软件架构与发布](chapters/11-软件架构与发布.md) | 分层架构、配置驱动管线、可复现发布流程 |
| VII 展望 | [12 进阶专题](chapters/12-进阶专题.md) | 能量反卷积、力匹配与量子嵌入、更高维动力学路线 |
| 附录 | [A 术语与公式速查](appendix/附录A-术语与公式速查.md) | 85 条术语 + 公式表 (标注代码位置) |
| | [B 常见陷阱与练习](appendix/附录B-常见陷阱与练习.md) | 本项目真实踩过的坑 + 15 道分级练习 |

## 使用方式

- 按章阅读: 公式 ($$...$$) 需要支持 LaTeX 的 Markdown 阅读器 (Typora/Obsidian/VS Code); 配图 (PNG) 直接内嵌。
- **配图全部由真实计算生成**: `python scripts/make_book_figures.py` 重现 `book/figures/` 中全部 14 张图 (无装饰性示意图; 概念图在标题中明确标注 schematic); 图中坐标轴为英文, 中文说明在正文图注。
- 动手实验可直接复制运行, 依赖仅 numpy/matplotlib/scipy:

```bash
pip install -e .
python autoquantum/examples/h3_wavepacket.py
python scripts/compare_qct_quantum.py --smoke
python -m unittest discover -s tests
```

- 书中所有 API、公式与基准数值均对照仓库源码核验; 未实现/实验性部分均有明确标注。
- 排版为单册: 运行 `python scripts/md2tex_book.py && cd book/build && xelatex book.tex && xelatex book.tex` 即可编译完整 PDF。

## 版本对应

本书对应 AutoQuantum v0.20.0 (全书 15 章、2 个附录、15 张真实计算配图 (含 He*+Li 集群实测))。核心验证数值 (波包 vs 解析基准、力训练提升、double-backprop FD 门控、QCT 能量守恒与隧穿差值) 见仓库 [RELEASE_VALIDATION.md](../RELEASE_VALIDATION.md)。
