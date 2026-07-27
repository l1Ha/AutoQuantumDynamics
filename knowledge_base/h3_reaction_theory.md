# H + H₂ 反应动力学知识库

## 1. 三原子反应体系

最简单的化学反应: H + H₂ → H₂ + H

### 1.1 坐标系统

**共线反应 Jacobi 坐标:**
- $R$: H 原子到 H₂ 质心的距离 (反应坐标)
- $r$: H₂ 键长 (振动坐标)
- $\theta$: 夹角 (旋转坐标, 共线时为 0)

### 1.2 LEPS 势能面

London-Eyring-Polanyi-Sato (LEPS) 势:

$$V = Q_{AB} + Q_{BC} + Q_{AC} - \sqrt{\frac{1}{2}[(J_{AB}-J_{BC})^2 + (J_{BC}-J_{AC})^2 + (J_{AC}-J_{AB})^2]}$$

其中:
- $Q(r) = \frac{1}{2}[E_M(r) + E_{AM}(r)]$: Coulomb 积分
- $J(r) = \frac{1}{2}[E_M(r) - E_{AM}(r)]$: 交换积分
- $E_M(r)$: Morse 势
- $E_{AM}(r)$: 反 Morse 势

### 1.3 H + H₂ 反应特性

| 参数 | 值 (au) | 说明 |
|------|---------|------|
| 势垒高度 | ~0.016 | 反应活化能 |
| 势垒位置 | R≈2.8, r≈1.8 | 鞍点 |
| 反应焓变 | 0 | 对称反应 |

## 2. 二维量子散射

### 2.1 Hamiltonian

$$H = -\frac{1}{2\mu_R}\frac{\partial^2}{\partial R^2} - \frac{1}{2\mu_r}\frac{\partial^2}{\partial r^2} + V(R, r)$$

约化质量:
- $\mu_R = \frac{2}{3}m_H$ (R 坐标)
- $\mu_r = \frac{1}{2}m_H$ (r 坐标)

### 2.2 数值求解

- 有限差分: 二阶中心差分
- Numerov 方法: 高精度递推
- ABC (吸收边界条件): 复势吸收
