# 附录 B 常见陷阱与练习

## B.1 陷阱清单

下面的“回归测试”优先采用仓库已有测试名；标为“建议新增”的项目目前没有专门门控，不能把文档中的名字误当成现有 API。

| 症状 | 根因 | 修复 | 回归测试名 |
|---|---|---|---|
| 谐振子基态看起来被挤压、能级/位置漂移 | 把振幅宽度 $\sigma$ 当成位置标准差；$|\psi|^2\propto e^{-(\Delta x/\sigma)^2}$ 的位置 SD 是 $\sigma/\sqrt2$ | 使用 `harmonic_ground_width` 的 $\sigma=1/\sqrt{\mu\omega}$，不要再次乘/除 $\sqrt2$ | `test_ground_state_is_stationary` |
| 2D 波包 CAP 放置后精度下降或分裂不再二阶 | CAP 只在完整时间步末附加；半步势没有吸收，且重构顺序破坏 Strang 对称性 | 构造 $W=V-i\eta$，在 `W/2 → T → W/2` 两侧都吸收，并按格点归因损失 | `test_absorption_conserves_total_probability`（可补充二阶收敛测试） |
| 初始布居被“凭空”损失，结果低于真实概率 | CAP 覆盖初始波包；初始波包在开始时已经处于阻尼区 | 先调用 `check_initial_overlap`；重叠 $>10^{-4}$ 报错、$>10^{-5}$ 告警，扩大网格或减小 `cap_width_frac` | `test_initial_overlap_check` |
| NumPy 2 环境在积分或初始化时报 `np.trapz` 不存在 | NumPy 2 移除了旧名，推荐 API 是 `np.trapezoid` | 使用 `autoquantum.dynamics.wavepacket.trapezoid` 的兼容包装；新代码优先写 `np.trapezoid` | `test_initialize_normalized_complex`（增加 NumPy 2 矩阵） |
| 1D `WavePacket1D.initialize` 随后乘相位时报实数/复数错误 | 初值曾为实数，乘 `exp(i phase)` 试图原位写复数 | 初始化后显式 `.astype(complex)`，再做复指数乘法 | `test_initialize_normalized_complex` |
| 禁阻区递推出现非物理振荡/结果被裁成看似正常 | `quantum_2d` 中心差分递推把 $2-h^2k^2$ 写成 $2+h^2k^2$；裁剪又掩盖错误 | 使用 $2-h^2k^2$，允许 $k^2<0$ 的虚指数；不要用 `[0,1]` 裁剪掩盖无效输出 | 建议新增 `test_quantum_2d_recurrence_sign` |
| 1D 波包向左传播时“透射/反射”曲线互换 | 方向由 `p0` 决定：从右侧向左时左侧是透射区，engine 的 CAP 标签曾按相反方向理解 | 透射=越过垒心区域+左 CAP；反射=返回区域+右 CAP，并用 $p_0$ 检查 | 建议新增 `test_1d_wavepacket_labels_follow_momentum` |
| 2D 扫描把 $E=p_R^2/(2\mu_R)$ 解释成总能量，阈值位置错误 | `WavePacket2DScan` 文档规定的是初始平动碰撞能，不含振动零点能 | 读取 `energy_*` 语义并在图、表中标注“collision energy”；需要总能量时另加 $E_{\rm vib}$ | 建议新增 `test_energy_scan_uses_collision_energy` |
| 通道和不再为 1，掩码看似合理却没有报错 | product/reactant 掩码重叠，或两者没有覆盖整个网格；当前只对重叠显式抛错，完整划分才是恒等式条件 | 用互补条件 `$R<2.0$` 与 `$R\ge2.0$`；重叠抛 `ValueError`，空隙用覆盖测试发现 | 建议新增 `test_complementary_masks_conserve_channels` |
| 拟合 PES 无论怎样训练都落在约 `[-1,1]`，物理能量超界 | 输出层误用 tanh；有界激活只能用于隐藏层 | `FeedForwardNN.forward` 保持输出线性；`test_linear_output_layer` 固定预激活 3.0，`test_fit_outside_tanh_range` 覆盖区间外目标 | `test_linear_output_layer` / `test_fit_outside_tanh_range` |
| 梯度在不同机器/网格上忽大忽小，NN 力训练不稳定 | FD 的 $h$ 太小会放大舍入误差，太大则有 $O(h^2)$ 截断误差；`grad_h=1e-5` 不是普适常数 | 按能量尺度和函数曲率选 $h$，至少做两种步长/Richardson 检查；同时验证解析梯度 | `test_force_gradient_backward_fd` |
| 安装包中出现 `torch`，但模型完全由 NumPy 实现 | 旧依赖表残留，代码没有 torch 导入；`requirements.txt` 当前仅列 numpy/matplotlib/scipy | 删除未使用依赖并检查构建元数据；纯 NumPy NN 是当前验证策略 | 建议新增 `test_nn_dependency_surface` |
| `sync_github.sh` 同步到错误项目，甚至要求改 SSH 配置 | 脚本硬编码旧 remote `anfax/AutoQuantum`，并在无 `gh` 时生成 key、修改 `$HOME/.ssh/config` | 禁止在科研目录直接运行；改为显式传入 remote、先 dry-run、先备份/审阅 SSH 配置 | 建议新增 `test_sync_script_target_is_current` |
| 加载别人发来的 `.pkl` 模型即执行任意代码 | `FeedForwardNN.load` 使用 `pickle.load`，反序列化不是数据解析 | 只加载可信来源并校验哈希；长期方案是改为带 schema 的纯数组格式或受限加载器 | 建议新增 `test_model_load_rejects_untrusted` |

## B.2 练习（附简答）

1. **[入门] 单位换算。** 将 $0.015\,E_h$、$1.401\,a_0$ 和 $1836.15\,m_e$ 写成 SI 近似，并说明为何代码常用 au。**答：** 依次约为 $6.54\times10^{-20}$ J、$7.41\times10^{-11}$ m、$1.67\times10^{-27}$ kg；au 消除常数，交换成本低。
2. **[入门] 宽度。** 给出振幅高斯 $\psi\propto e^{-\Delta x^2/(2\sigma^2)}$ 的位置 SD、动量 SD 和不确定关系乘积。**答：** SD 分别为 $\sigma/\sqrt2$、$1/(\sqrt2\sigma)$，乘积为 1/2；不要把 $\sigma$ 当位置 SD。
3. **[入门] Morse/谐振。** 由 $V=D(1-e^{-\alpha(r-r_0)})^2$ 证明 $r_0$ 是势能最低点；计算小振动频率。**答：** $V'(r_0)=0,V''(r_0)=2D\alpha^2$；$\omega=\alpha\sqrt{2D/\mu}$。
4. **[入门] Numerov。** 推导一维无势、均匀网格的递推，并说明 $k^2<0$ 的物理意义。**答：** 使用 $2-h^2k^2$ 系数；负 $k^2$ 给出指数衰减/增长，允许描述禁阻区，不能简单裁掉。
5. **[入门] CAP。** 解释 `W=V-iη` 半步因子为何产生阻尼，并计算一次半步的振幅/概率变化。**答：** 因子含 $e^{-\eta dt/2}$，概率乘 $e^{-\eta dt}$；损失必须从概率差记账。
6. **[进阶] 通道恒等式。** 对互补掩码证明 $P_{\rm react}+P_{\rm refl}=1$；指出 CAP 角点重叠时如何保证确定性。**答：** 网格布居与全部累计吸收分割总概率；角点按最大 $\eta$ 的单一边归因。
7. **[进阶] 能量语义。** 从 `WavePacket2DScan.run` 推导中心动量，并说明为何振动基态能量不自动进入 `energy`。**答：** $p_{R0}=-\sqrt{2\mu_RE}$；`energy` 是平动碰撞能，基态振动能只在波包初态中隐含。
8. **[进阶] 波包卷积。** 对宽度 $\sigma$ 写出 $\tilde a(k)$，并比较中心能量 $T(E_0)$ 与波包结果。**答：** $|\tilde a|^2\propto e^{-\sigma^2k^2}$，结果为高斯动量分布对 $T(k^2/2\mu)$ 的加权平均；阈值附近差异很大。
9. **[进阶] 力训练。** 推导 $L=\mathrm{MSE}_V+w\mathrm{MSE}_F$ 对网络权重的一阶依赖，并说明为何需要 $f''$。**答：** 先对输入梯度反向，再对前向图反向；$f'(z)$ 对前向 $z$ 的导数含 $f''(z)$。
10. **[进阶] FD 误差。** 估计中心差分的 $O(h^2)$ 截断和 $O(\epsilon/h)$ 舍入项，说明如何选 $h$。**答：** 在两者交叉附近取 $h$；用 $h,h/2$ 或 Richardson 检查，避免把噪声当梯度。
11. **[进阶] 主动学习。** 为一维势能设计 D-optimal 与 committee disagreement 两步选点策略。**答：** 先用信息矩阵选覆盖，再在预测方差高的区域加密；每轮重训并检查验证误差和动力学敏感性。
12. **[专家] 反卷积。** 写出多宽度分段基的正规方程，并设计一个验证 $\lambda$ 的实验。**答：** $A_j=\int|\tilde a_j|^2\phi_n$，解 $(A^\mathsf{T}WA+\lambda I)c=A^\mathsf{T}WP$；用独立宽度/解析 $T$ 选择 $\lambda$。
13. **[专家] 2D 动力学。** 对无耦合 Eckart 体系说明为什么可与 1D 解析谱加权比较，并设计传播时间收敛测试。**答：** $r$ 方向保持谐振子基态，问题约化为 R；逐步增大 $n\_steps$ 直到反应概率、存活+吸收均稳定。
14. **[专家] 对称性。** 区分 SE(3)、E(3) 与置换不变性，设计一个 H₃ 交换测试。**答：** 旋转/平移应不改变能量，交换氢只交换通道标签；对 `(R,r)` 做 Jacobi 变换并检查同一物理 observables。
15. **[专家] 软件设计。** 为 `AbInitioData` 增加主动学习闭环时，给出最小 API、误差记录和防回归门控。**答：** 保持 `points/energies/gradients` 对齐；新增候选评分、批次合并、版本化配置；用解析函数、FD 梯度和守恒测试三层门控，不把未实现方法伪装成现有 API。

练习完成后建议按“能量→PES→波包→守恒/收敛”的顺序补测试；术语定义见附录A，理论与实现边界见“11 进阶专题”。
