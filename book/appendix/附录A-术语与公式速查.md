# 附录 A 术语与公式速查

本附录使用 AutoQuantum 的约定：$\hbar=1$；能量为 Hartree（$E_h$）；长度为 Bohr（$a_0$）；质量以电子质量 $m_e$ 为单位，氢原子取 $m_H=1836.15$。术语的“仓库用法”优先于泛化定义；尤其要区分总能量、碰撞能和波包中心能量。

## A.1 双语术语表

| 编号 | 中文 | English | 速查定义/仓库语境 |
|---:|---|---|---|
| 1 | 波包 | wave packet | 局域的波函数分布，具有有限位置和动量宽度。 |
| 2 | 势能面 | potential energy surface (PES) | 电子能量作为核坐标函数的曲面；仓库有 Morse、Harmonic、LJ、LEPS、Eckart。 |
| 3 | 波恩–奥本海默近似 | Born–Oppenheimer approximation | 冻结电子运动，把电子态能量作为核 PES。 |
| 4 | 约化质量 | reduced mass | 相对运动的等效质量；H+H₂ 中 $\mu_R=2m_H/3$、$\mu_r=m_H/2$。 |
| 5 | Jacobi 坐标 | Jacobi coordinates | 把反应坐标与分子内坐标分离；本仓库用 $(R,r)$。 |
| 6 | 量子散射 | quantum scattering | 研究入射波在势垒/势面上的透射、反射和反应。 |
| 7 | 定态散射 | stationary scattering | 在给定能量求解定态薛定谔方程；`quantum_1d.py` 为 Numerov 路径。 |
| 8 | 含时动力学 | time-dependent dynamics | 传播 $\psi(t)$；仓库使用 Split-Operator FFT。 |
| 9 | 薛定谔方程 | Schrödinger equation | $i\hbar\partial_t\psi=(T+V)\psi$。 |
| 10 | 哈密顿量 | Hamiltonian | 动能与势能算符之和。 |
| 11 | 波函数 | wave function | 复振幅 $\psi$，通常以离散网格或 FFT 系数存储。 |
| 12 | 概率密度 | probability density | $|\psi|^2$；二维积分使用 $\Delta R\Delta r$。 |
| 13 | 概率流 | probability flux | 通道结果的连续性对应量；当前二维结果主要用掩码与 CAP 记账。 |
| 14 | 幺正性 | unitarity | 无吸收时传播保持内积/概率；一维、二维测试均检查。 |
| 15 | 吸收势 | absorbing potential | 用复势 $-i\eta$ 阻尼出射波，避免 FFT 周期回绕。 |
| 16 | 复吸收势 | complex absorbing potential (CAP) | `W=V-i\eta`，边缘二次型 ramp。 |
| 17 | 分裂算符 | split-operator method | 把动力学算符拆为动能和势能子步传播。 |
| 18 | Strang 分裂 | Strang splitting | `W/2 → T → W/2` 的二阶对称时间分裂。 |
| 19 | Numerov 方法 | Numerov method | 一维定态薛定谔方程的高阶中心递推。 |
| 20 | 透射 | transmission | 入射概率中越过势垒进入透射通道的比例。 |
| 21 | 反射 | reflection | 返回入射渐近区的概率比例。 |
| 22 | 透射率 | transmission probability | 单能散射的 $T$ 或波包累计的 $P_{\rm trans}$，需看扫描语义。 |
| 23 | 反应概率 | reaction probability | 产物区布居加产物 CAP 吸收的有限时间估计。 |
| 24 | 通道分析 | channel analysis | 按几何掩码、CAP 边统计反应/反射通道。 |
| 25 | 产物通道 | product channel | LEPS 中 $r_{AB}<r_{BC}$ 的交换后区域或对应 CAP。 |
| 26 | 反应物通道 | reactant channel | 初始入射区域；Eckart 常用 $R\ge R_{\rm div}$。 |
| 27 | 吸收通道 | absorbing channel | CAP 边上累计吸收概率。 |
| 28 | 通道恒等式 | channel identity | 互补掩码时 $P_{\rm react}+P_{\rm refl}=1$。 |
| 29 | 概率守恒恒等式 | norm identity | $P_{\rm grid}+\sum_eL_e=1$。 |
| 30 | 阈值 | threshold | 进入开放反应通道所需的最低碰撞能；LEPS 教学阈值不是真实物理量。 |
| 31 | 共振 | resonance | 有限寿命的准束缚态，常造成能量依赖的透射峰。 |
| 32 | 能谱 | energy spectrum | Hamiltonian 特征能量或波包动量分布。 |
| 33 | 能量分辨率 | energy resolution | 波包有限宽度导致的 $\Delta E$。 |
| 34 | 碰撞能 | collision energy | 仓库 `WavePacket2DScan` 的 $E=p_R^2/(2\mu_R)$。 |
| 35 | 总能量 | total energy | 相对平动、振动、电子等全部自由度；不等同于扫描的碰撞能。 |
| 36 | 初始平动能 | initial translational energy | 仓库 1D 管线由选定波包中心动量近似得到的能量。 |
| 37 | 初始动量 | initial momentum | `WavePacket2D.p_R0` 或 `WavePacket1D.p0`。 |
| 38 | 动量分布 | momentum distribution | 波函数的 Fourier 振幅平方 $|\tilde a(p)|^2$。 |
| 39 | 能量卷积 | energy convolution | $P(E)=\int|\tilde a(E')|^2T(E')dE'$。 |
| 40 | 反卷积 | deconvolution | 从多宽度卷积观测估计单能 $T(E)$。 |
| 41 | 冲激响应 | impulse response | 对理想窄能量分布的线性响应；不是累计概率曲线。 |
| 42 | 累计概率 | cumulative probability | 随时间单调累加的通道/CAP 吸收量。 |
| 43 | T2P | time-to-potential | Baer–Neuhauser 从波函数谱分量反演通道响应的路线。 |
| 44 | 渐近区 | asymptotic region | 入射或出射波已形成平面波/波包区。 |
| 45 | 分界面 | dividing surface | 区分反应物和产物的几何线或面。 |
| 46 | 初始态 | initial state | 传播开始前的波包；2D 用基态宽度或谐振近似初始化。 |
| 47 | 基态宽度 | ground-state width | 谐振子 $\sigma=1/\sqrt{\mu\omega}$，在仓库中是振幅宽度。 |
| 48 | 谐振子 | harmonic oscillator | $V=\tfrac12k(x-x_0)^2$，常用于基态和可分离性测试。 |
| 49 | Morse 势 | Morse potential | $D[1-e^{-\alpha(r-r_0)}]^2$，双原子解析面。 |
| 50 | Eckart 势 | Eckart potential | $V_0/\cosh^2(\beta A)$ 的简化 H+H₂ 势垒模型。 |
| 51 | LEPS 势 | London–Eyring–Polanyi–Sato potential | 用 $Q,J$ 组合描述三原子反应的解析面。 |
| 52 | 代理面 | surrogate PES | NN 或其它低成本函数对高成本 PES 的替代。 |
| 53 | 有限差分 | finite difference (FD) | 用邻近点估计数值/导数。 |
| 54 | 中心差分 | central difference | $[f(x+h)-f(x-h)]/(2h)$；`sample_function` 用它生成梯度。 |
| 55 | 差分步长 | finite-difference step $h$ | 截断误差与舍入误差的折中；当前默认 `grad_h=1e-5`。 |
| 56 | 解析梯度 | analytic gradient | 由计算/自动微分一次得到的导数；仓库 NN 已有 `gradient`。 |
| 57 | 力训练 | force training | 用 $\partial V/\partial x$ 监督并联合优化能量误差。 |
| 58 | double backprop | double backpropagation | 对输入梯度再反传，获得力损失的参数梯度。 |
| 59 | Adam | Adam optimizer | 带一阶、二阶动量的梯度优化器；`NNTrainer` 内置。 |
| 60 | 早停 | early stopping | 验证损失无相对改进时停止，并恢复最优权重。 |
| 61 | 标准化 | standardization | 用训练划分的均值/标准差缩放输入输出，并持久化。 |
| 62 | 恒等式 | identity | 解析或数值应严格/近似满足的守恒关系。 |
| 63 | 收敛 | convergence | 迭代误差或物理量随网格/时间/训练减小。 |
| 64 | 离散化误差 | discretization error | 有限差分、网格、FFT、时间步造成的误差。 |
| 65 | 混叠 | aliasing | 采样不足导致不同频率折叠；需 CAP、滤波或更细网格。 |
| 66 | Nyquist 频率 | Nyquist frequency | 离散采样可表示的最高频率，约为 $\pi/\Delta x$。 |
| 67 | D-最优性 | D-optimality | 最大化设计矩阵对数行列式的采样准则。 |
| 68 | GHOST/VDB 集成 | GHOST/VDB ensemble | 基于模型分歧/偏差的主动学习采样概念。 |
| 69 | 委员会分歧 | committee disagreement | 多个 NN 预测方差作为不确定性代理。 |
| 70 | 力匹配 | force matching | 最小化代理力与参考力差异。 |
| 71 | $\lambda$-拟合 | $\lambda$-fitting | 在能量与力损失间以 $\lambda$ 显式折中。 |
| 72 | QISA | quantum integrated algorithm | 用辅助经典路径积分表达量子动力学的嵌入框架。 |
| 73 | $\Delta$-learning | delta learning | 学习高精度模型与低精度模型之间的残差。 |
| 74 | 置换对称性 | permutational symmetry | 同类原子交换后物理 observables 不变。 |
| 75 | SE(3)/E(3) 不变性 | SE(3)/E(3) invariance | 对三维旋转/平移（E(3) 还含反射）不变。 |
| 76 | 置换不变性 | permutation invariance | 交换同类原子输入标签，标量输出不变。 |
| 77 | Behler–Parrinello 对称函数 | Behler–Parrinello symmetry functions | 用原子对距离的高斯径向/角向基描述局域环境；本仓库尚未实现。 |
| 78 | 张量列 | tensor train (TT) | 以低秩链压缩高维张量；本仓库尚未实现。 |
| 79 | 无网格方法 | grid-free method | 以高斯、DVR 等表示连续空间，绕开规则全维网格。 |
| 80 | Gauss–Hermite 求积 | Gauss–Hermite quadrature | 对高斯权重积分高效；可作低维波函数基。 |
| 81 | DVR | density value representation | 用连续函数值离散微分算符的高阶基。 |
| 82 | Chebyshev 稀疏化 | Chebyshev sparsification | 用正交多项式/稀疏截断减少状态数。 |
| 83 | 模式选择性 | mode selectivity | 指定反应路径或振动模式主导反应通道。 |
| 84 | 曲率坐标 | curvilinear coordinates | 沿反应路径及正交模式重写 Hamiltonian。 |
| 85 | Feynman 路径积分 | Feynman path integral | 用路径历史求量子传播的另一种表述。 |

## A.2 公式速查

| 主题 | 公式 | 仓库中的使用位置/含义 |
|---|---|---|
| 单位约定 | $\hbar=1,\ E_h,\ a_0,\ m_H=1836.15$ | `wavepacket_2d.py` 顶部注释；H+H₂ 质量由 `h3_reduced_masses` 返回。 |
| H+H₂ 约化质量 | $\mu_R=2m_H/3,\ \mu_r=m_H/2$ | `h3_reduced_masses`；$(R,r)$ 动能系数。 |
| 高斯振幅包 | $\psi(x)\propto e^{-\frac12(\Delta x/\sigma)^2}e^{ip_0\Delta x}$ | `WavePacket1D.initialize`、`WavePacket2D.initialize`。 |
| 振幅宽度与位置标准差 | $|\psi|^2\propto e^{-(\Delta x/\sigma)^2},\ \mathrm{SD}_x=\sigma/\sqrt2$ | 2D docstring 与 `test_ground_state_is_stationary`；σ 不是位置 SD。 |
| 动量标准差与能量分辨率 | $\mathrm{SD}_p=1/(\sqrt2\sigma),\ \Delta E\simeq\mu\,\mathrm{SD}_p^2=1/(4\mu\sigma^2)$ | `WavePacket2D` docstring；解释波包卷积宽度。 |
| k 空间动能 | $T(k)=\frac12(g_{RR}k_R^2+2g_{Rr}k_Rk_r+g_{rr}k_r^2)$，对角时 $g_{ii}=1/\mu_i$ | `WavePacket2DPropagator.__init__` 的 `T_k`；`g_matrix` 支持非对角项。 |
| Strang 分裂 | $U(dt)\approx e^{-iWdt/2}\mathcal F^{-1}[e^{-iTdt}\mathcal F(e^{-iWdt/2}\psi)]$ | `WavePacket2DPropagator.step`；$W=V-i\eta$。 |
| CAP 阻尼因子 | $e^{-i(V-i\eta)dt/2}=e^{-iVdt/2-\eta dt/2}$ | 2D 在两个半步都施 CAP；不能用步末一次性附加阻尼替代。 |
| 一维透射/反射 | $T=\frac{k_t}{k_i}|A_t/A_i|^2,\ R=|A_r/A_i|^2$ | `QuantumScattering1D` 的匹配边界；能量扫描为单能结果。 |
| 透射守恒 | $T+R=1$ | 1D 传播与散射的检查；CAP 存在时改查存活+吸收。 |
| 通道记账 | $P_{\rm react}=P_{\rm prod}+\sum_{e\in{\rm product}}L_e$；$P_{\rm refl}=P_{\rm reactant}+\sum_{e\notin{\rm product}}L_e$ | `WavePacket2DPropagator.propagate`；掩码互补时二者之和为 1。 |
| 波包能量卷积 | $P(E)=\int|\tilde a(E')|^2T(E')dE'$ | `WavePacket2DScan` 的正确解释；测试用能量加权 1D 参考。 |
| 碰撞能扫描 | $E=p_{R0}^2/(2\mu_R)$ | `WavePacket2DScan.run`；不是总能量。 |
| 谐振子基态宽度 | $\omega=\sqrt{k/\mu},\ \sigma=1/\sqrt{\mu\omega}$ | `harmonic_ground_width`；返回振幅宽度。 |
| Morse 近似宽度 | $\omega=\alpha\sqrt{2D/\mu},\ \sigma\simeq1/\sqrt{\mu\omega}$ | `morse_ground_width`；是谐振近似，不是精确 Morse 宽度。 |
| NN 能量/力损失 | $L=\mathrm{MSE}(V,\hat V)+w\,\mathrm{MSE}(\nabla V,\nabla\hat V)$ | `NNTrainer.combined_loss`；$w=force_weight$，默认 2D 管线为 1。 |
| 中心差分梯度 | $\partial_jf(x)\simeq[f(x+h e_j)-f(x-h e_j)]/(2h)$ | `AbInitioData.sample_function`；当前 `grad_h=1e-5`。 |
| 神经解析梯度 | $\nabla_x\hat y=\frac{y_{\rm scale}}{x_{\rm scale}}\nabla_{x_{\rm std}}\hat y_{\rm std}$ | `FeedForwardNN.gradient`；以有限差分测试。 |
| double-backprop 伴随 | $\Phi=\sum_{ij}\bar G_{ij}(\nabla_x\hat y)_{ij}$，再对 $\Phi$ 反传 | `input_gradient_backward`；力训练需要 $f''$。 |
| 分段基反卷积 | $T(E)\approx\sum_n c_n\phi_n(E)$，$\min_c\sum_jw_j\|A_jc-P_j\|^2+\lambda\|c\|^2$ | 10.1 的未实现多宽度方案；不是 `WavePacket2DScan` 的隐含功能。 |
| 概率守恒 | $\sum_i|\psi_i|^2\Delta x+\sum_eL_e=1$ | CAP 测试；二维使用 $\Delta R\Delta r$。 |
| 误差与采样 | $\mathrm{RMSE}=\sqrt{N^{-1}\sum_i(y_i-\hat y_i)^2}$ | engine 对 NN 面报告 `nn_fit_rmse`、梯度 RMSE；动力学不能只看训练 MSE。 |

## A.3 电子结构术语补充 (第 4 章)

| 术语 (中/英) | 含义 | 本仓库关联 |
|---|---|---|
| 基组 / basis set | 用有限个高斯基函数线性组合展开分子轨道; `sto-3g`, `cc-pVDZ` | `PySCFCalculator(basis=...)` |
| 高斯基函数 / Gaussian basis function | $\phi_{ijk}=N x^iy^jz^k e^{-\alpha r^2}$ 及其球谐组合 | 第 4 章 4.2 节 |
| 收缩基组 / contracted | 多个原语高斯基组合为一个"轨道"; STO-3G = 3 个高斯基近似一个 Slater 轨道 | 第 4 章 |
| 弥散函数 / diffuse function | 大 $\alpha$ 指数的小尾函数, 阴离子/长程相互作用必需 | 第 4 章 7 节 |
| BSSE / 基组叠加误差 | 用有限单体外壳计算二聚体时人为降低结合能 | 采样数据解读; 见第 4 章 |
| Slater 行列式 / Slater determinant | 电子波函数反对称化的行列式表示 | 第 4 章 4.1 节 |
| 变分原理 | $\delta\langle H\rangle=0$ 给出基组内最优波函数 | 第 4 章 4.1 节 |
| 哈特里-福克 / Hartree–Fock (HF) | 单 Slater 行列式的自洽场方法, 精确交换 | `method='rhf'/'uhf'/'rohf'` |
| Fock 矩阵 / Fock matrix | $F=h+G$, Roothaan–Hall 的核心算符矩阵 | 第 4 章 4.3 节 |
| Roothaan–Hall 方程 | $FC=SC\varepsilon$ 广义特征值问题, SCF 迭代求不动点 | `PySCFCalculator._mf` |
| 自洽场 / SCF | 轨道→密度→Fock→轨道 循环至收敛 | `mf.converged` 检查 |
| 密度泛函 / DFT | 用电子密度代替波函数; Hohenberg–Konn 定理 | `method='dft'` |
| Kohn–Sham 方程 | 用辅助非相互作用体系在有效势下求解 | 第 4 章 4.4 节 |
| 交换-相关泛函 / XC functional | $E_{xc}[n]$ 的近似 (LDA/GGA); 直接影响 NN 标签 | provenance 记录 |
| 半经验方法 / semi-empirical | 参数化的电子哈密顿量 (GFN-xTB), 大体系快速 | `XTBCommandCalculator` |
| xTB | 扩展紧束缚半经验方法; `--chrg/--uhf/--acc` 参数 | `XTBCommandCalculator.__init__` |
| Hellmann–Feynman | 基组固定时 $\partial E/\partial R=\langle\psi|\partial H/\partial R|\psi\rangle$ | 梯度来源; `energy_and_gradient` |
| Pulay 力 | 基组依赖时需变分导数修正; Gaussian 程序自动处理 | 第 4 章高手专栏 |
| 核排斥能 / nuclear repulsion | 几何能量中的 $Z_AZ_B/R_{AB}$ 项, 与电子能合并才是总能量 | 梯度与力的定义 |
| 电荷/多重度 / charge, multiplicity | 电子数与自旋态; 填错会得到错误势能面 | `--charge/--uhf/spin` |
| 力 vs 梯度 | $\mathbf{F}=-\nabla_R E$ (ASE 返回力, 本项目取负号得梯度) | `ASECalculatorAdapter` |
| 零点能 / ZPE | 谐振零点能, **不在**电子能量/力中; 振动态 PES 需另行加和 | NN 面继承纯电子面 |

相关章节：电子结构见“04 电子结构基础”；数学与单位见“01 数学预备”；波包实现和风险见“附录B”；概念定义和公式来源可回看 `knowledge_base/*.md`。
