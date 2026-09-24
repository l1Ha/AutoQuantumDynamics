# AutoQuantum 教程与理论手册

从量子力学基础到可验证的分子反应动力学计算——与 [AutoQuantum](../) 代码库逐行对应的中文书籍。

## 适读人群

- **新手**: 有微积分与基础物理背景, 第一次接触量子散射/分子动力学。第 1-3 章是入口, 每章都有"新手路径"专栏。
- **进阶者**: 要读懂或改造本仓库代码的开发者。建议按 4 → 6 → 8 章顺序阅读。
- **高手**: 关心算法选择、数值稳定性与验证严格性的研究者。各章"高手专栏"、第 8/10 章与附录 A/B 面向此层。

## 章节结构

| 部分 | 章节 | 内容 |
|---|---|---|
| 序 | [00 前言与学习路径](chapters/00-前言与学习路径.md) | 本书缘起、三条学习路线、符号与单位约定 |
| I 基础 | [01 数学预备](chapters/01-数学预备.md) | 复数空间、DFT/卷积定理、采样与 Nyquist、差分与求积 |
| | [02 量子力学基础](chapters/02-量子力学基础.md) | Born-Oppenheimer、定态/含时方程、期望值、原子单位 |
| II 势能面 | [03 势能面](chapters/03-势能面.md) | 键坐标 vs Jacobi 坐标、约化质量、Morse/LEPS/Eckart、数据接口 |
| III 电子结构 | [04 电子结构基础](chapters/04-电子结构基础.md) | Born-Oppenheimer 变分求解、基组与高斯基函数、SCF/Roothaan、DFT (LDA/GGA)、半经验 xTB、梯度与力 |
| IV 机器学习 | [05 神经网络势能面](chapters/05-神经网络势能面.md) | 反向传播、Adam、标准化、力训练与双反向传播 |
| V 动力学 | [06 定态散射](chapters/06-定态散射.md) | 匹配条件、透射/反射、Numerov、实验性 2D 求解器 |
| | [07 含时波包](chapters/07-含时波包.md) | 高斯波包、分裂算符、CAP、概率记账恒等式 |
| | [08 二维反应动力学](chapters/08-二维反应动力学.md) | 交换分界面、能量平均透射率、阈值行为 |
| VI 工程 | [09 数值验证](chapters/09-数值验证.md) | 收敛研究、解析基准、有限差分门控、恒等式测试、仲裁工作流 |
| | [10 软件架构与发布](chapters/10-软件架构与发布.md) | 分层架构、配置驱动管线、可复现发布流程 |
| VII 展望 | [11 进阶专题](chapters/11-进阶专题.md) | 能量反卷积、主动学习、力匹配、对称性、GPU 路线 (多数未实现) |
| 附录 | [A 术语与公式速查](appendix/附录A-术语与公式速查.md) | 85 条术语 + 公式表 (标注代码位置) |
| | [B 常见陷阱与练习](appendix/附录B-常见陷阱与练习.md) | 本项目真实踩过的坑 + 15 道分级练习 |

## 使用方式

- 按章阅读: 公式 ($$...$$) 需要支持 LaTeX 的 Markdown 阅读器 (Typora/Obsidian/VS Code); 配图 (PNG) 直接内嵌。
- **配图全部由真实计算生成**: `python scripts/make_book_figures.py` 重现 `book/figures/` 中全部 12 张图 (无装饰性示意图; 概念图在标题中明确标注 schematic); 图中坐标轴为英文, 中文说明在正文图注。
- 动手实验可直接复制运行, 依赖仅 numpy/matplotlib/scipy:

```bash
pip install -e .
python autoquantum/examples/h3_wavepacket.py
python -m unittest discover -s tests
```

- 书中所有 API、公式与基准数值均对照仓库源码核验; 未实现/实验性部分均有明确标注。
- 排版为单册: 约定 `pandoc book/chapters/*.md book/appendix/*.md -o book.pdf` 可合并为 PDF (需自行安装 pandoc 与中文字体)。

## 版本对应

本书对应 AutoQuantum v0.9.0 (第 3.6 节含从头算后端与数据生成; 第 4 章电子结构基础; 第 6 章分子振动基础; 全书 12 张真实计算配图)。核心验证数值 (波包 vs 解析基准、力训练提升、double-backprop FD 门控) 见仓库 [RELEASE_VALIDATION.md](../RELEASE_VALIDATION.md)。
