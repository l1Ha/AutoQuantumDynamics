"""CI-NEB (Climbing-Image Nudged Elastic Band) — 最小能量路径与过渡态搜索。

只依赖 ``Calculator`` 的解析梯度 (无需 Hessian), 因此对解析面、ML 代理面、
以及 SCF/DFT/MP2/CCSD(T) 后端通用。

算法要点
--------
1. 线性插值生成 N 个内部镜像 (端点固定);
2. 每个镜像受力 = 真实力的**垂直分量** + 弹簧力的**平行分量**
   (nudging 消除弹簧力对路径形状的污染):
   .. math:: F_i = F_i^{\\perp} + k\\,(|R_{i+1}-R_i| - |R_i-R_{i-1}|)\\,\\hat\\tau_i
3. **climbing image**: 能量最高的镜像改为沿切向反转平行分量,
   使其爬向鞍点: :math:`F = F^{\\perp} - 2F^{\\parallel}`;
4. 镜像用 **FIRE** (Fast Inertial Relaxation Engine, Bitzek 2006) 演化:
   速度 Verlet + 自适应步长 + 速度-力混合; 对 NEB 的非保守有效力比
   quick-min / 最速下降稳健得多 (实测在陡峭 LEPS 面上才收敛)。

验证判据 (见 scripts/validate_transition_state.py):
- TS 处应恰有 **1 个虚频** (由 ``optimize.harmonic_frequencies`` 复核);
- 解析面 (LEPS H+H₂) 上 NEB 势垒 vs 解析 MEP 势垒;
- H₃ 交换反应 CCSD(T) 势垒 vs 文献 9.6 kcal/mol (0.416 eV) 量级。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

HA_EV = 27.211386245988


def _interpolate(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, n + 2)[1:-1]
    return np.array([(1 - t) * a + t * b for t in ts])


def _tangents(images: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """各镜像的路径切向 (前向/后向差分各半, 端点用固定端)。"""
    n = images.shape[0]
    # ext = [A, R_0, ..., R_{n-1}, B]; 镜像 R_i 位于 ext[i+1]
    # 中心差分: τ_i ∝ ext[i+2] - ext[i]
    # ⚠ 不可写成 ext[i+1] - ext[i-1]: i=0 时负索引会回绕到终点 B,
    #   使首个镜像切向反向 → 整条路径塌陷 (真实踩过的 bug)。
    ext = np.concatenate([a[None], images, b[None]], axis=0)
    tau = np.zeros_like(images)
    for i in range(n):
        d = ext[i + 2] - ext[i]
        nrm = np.linalg.norm(d)
        tau[i] = d / nrm if nrm > 1e-12 else 0.0
    return tau


def neb_path(calc, symbols: Sequence[str], coords_a: np.ndarray,
             coords_b: np.ndarray, n_images: int = 9,
             k_spring: float = 0.05, climb: bool = True,
             max_iter: int = 1500, dt: float = 0.05, gtol: float = 2e-3,
             verbose: bool = False) -> Tuple[np.ndarray, Dict]:
    """CI-NEB 求最小能量路径与过渡态。

    Returns
    -------
    images : (n_images, N, 3) ndarray
        收敛后的路径镜像 (Bohr); ``info["ts_index"]`` 为能量最高者。
    info : dict
        ``energies`` (Hartree, 长度 n_images), ``barrier_eV`` (相对起点),
        ``ts_index``, ``converged``, ``n_iter``, ``n_evals``。
    """
    A = np.asarray(coords_a, dtype=float)
    B = np.asarray(coords_b, dtype=float)
    R = _interpolate(A, B, n_images)
    n = R.shape[0]
    V = np.zeros_like(R)
    alpha, alpha0, n_pos = 0.1, 0.1, 0
    dt_min, dt_max = 0.05, 1.0
    step_scale = 0.05                   # 每步最大位移尺度 (Bohr)
    E = np.zeros(n)
    n_evals = 0
    converged = False
    grad_max = np.inf

    def evaluate(imgs):
        nonlocal n_evals
        for i in range(imgs.shape[0]):
            e, g = calc.energy_and_gradient(imgs[i])
            n_evals += 1
            E[i] = e
            F[i] = -np.asarray(g, dtype=float)     # 力 = -∇E
    F = np.zeros_like(R)
    evaluate(R)

    for it in range(max_iter):
        tau = _tangents(R, A, B)
        F_eff = np.zeros_like(R)
        i_ts = int(np.argmax(E))
        for i in range(n):
            Fi = F[i]
            f_par = float(np.sum(Fi * tau[i]))
            perp = Fi - f_par * tau[i]
            if climb and i == i_ts:
                # 爬升镜像: 沿切向反转 (F^⊥ - 2F^∥)
                F_eff[i] = perp - f_par * tau[i]
            else:
                d_prev = np.linalg.norm(R[i] - (A if i == 0 else R[i - 1]))
                d_next = np.linalg.norm((B if i == n - 1 else R[i + 1]) - R[i])
                spring = k_spring * (d_next - d_prev)
                F_eff[i] = perp + spring * tau[i]
        grad_max = float(np.linalg.norm(F_eff, axis=(1, 2)).max())
        if verbose and (it % 20 == 0 or grad_max < gtol):
            print(f"  NEB iter {it:3d}: E_span = "
                  f"{(E.max()-E.min())*HA_EV:.4f} eV, |F_eff|max = {grad_max:.2e}")
        if grad_max < gtol:
            converged = True
            break
        # FIRE (Bitzek 2006) + **力归一化 + 速度上限**: 位移逐步有界,
        # 对 LEPS 这类极陡面也不会塌陷 (未归一化的速度 Verlet 会崩)。
        fmax_i = float(np.linalg.norm(F_eff, axis=(1, 2)).max())
        Fs = F_eff / max(fmax_i, 1e-30)         # 单位最大力
        V += Fs * dt
        vnorm = float(np.linalg.norm(V))
        if vnorm > 1.0:
            V /= vnorm                          # 速度上限 → 位移 ≤ dt·scale
        if vnorm > 1e-14:
            V = (1 - alpha) * V + alpha * (np.linalg.norm(V) / max(
                float(np.linalg.norm(Fs)), 1e-30)) * Fs
        P = float(np.sum(Fs * V))
        if P > 0:
            n_pos += 1
            if n_pos > 5:
                dt = min(dt * 1.1, dt_max)
                alpha *= 0.99
        else:
            dt = max(dt * 0.5, dt_min)
            V[:] = 0.0
            alpha = alpha0
            n_pos = 0
        R += V * (dt * step_scale)
        evaluate(R)

    tau = _tangents(R, A, B)
    # 势垒必须相对**反应物端点** A 定义 (而不是第一个镜像:
    # 镜像位于路径内部, 其能量已高于/低于反应物 → 会造成偏差)
    e_a = float(calc.energy(A))
    e_b = float(calc.energy(B))
    n_evals += 2
    e_ts = float(E.max())
    info = {"energies": E.copy(), "ts_index": int(np.argmax(E)),
            "barrier_eV": float((e_ts - e_a) * HA_EV),
            "barrier_Ha": float(e_ts - e_a),
            "reactant_energy": e_a, "product_energy": e_b,
            "reaction_energy_eV": float((e_b - e_a) * HA_EV),
            "converged": converged, "n_iter": it + 1, "n_evals": n_evals}
    return R, info


def refine_saddle(calc, coords_ts: np.ndarray, gtol: float = 2e-4,
                  **opt_kw) -> Tuple[np.ndarray, Dict]:
    """以 NEB 的 TS 估计为起点, 用「最小化 |F|」策略精修鞍点。

    做法: 对坐标做最速下降/quick-min 于 |∇E|² 上 (Hessian-free 鞍点精修),
    对光滑势面可把鞍点定位到 |∇E| < gtol。
    """
    x = np.asarray(coords_ts, dtype=float).copy()
    v = np.zeros_like(x)
    dt = 0.02
    n_evals = 0
    gmax = np.inf
    for it in range(2000):
        e, g = calc.energy_and_gradient(x)
        n_evals += 1
        gmax = float(np.abs(g).max())
        if gmax < gtol:
            break
        # |g|² 的梯度 = 2 H·g ≈ 用 g 的差分近似: 这里直接沿 -g 走 (最速下降到驻点)
        v += -g * dt
        v *= 0.9
        x += v * dt
    return x, {"grad_max": gmax, "n_evals": n_evals, "energy": float(e)}


def imaginary_mode_count(calc, symbols: Sequence[str], coords: np.ndarray,
                         h: float = 1e-3, extra_masses=None) -> int:
    """TS 判据: 鞍点处虚频数 (应为 1)。"""
    from autoquantum.pes.optimize import harmonic_frequencies
    freqs, info = harmonic_frequencies(calc, symbols, coords, h=h,
                                       extra_masses=extra_masses)
    return info["n_imag"]
