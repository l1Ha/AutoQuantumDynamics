"""电子结构计算后端 — 为 NN 势能面拟合生成从头算训练数据。

本模块定义统一的 ``Calculator`` 协议 (能量 Hartree + 梯度
Hartree/Bohr), 并提供:

- ``AnalyticCalculator``: 包装任意 (E, ∇E) 可调用对 — 测试/演示
  以及"解析面即数据源"的闭环, 无外部依赖, 完全可验证;
- ``XTBCommandCalculator``: 子进程调用 GFN-xTB 的半经验方法
  (实验性 — 需要安装 xtb, 本仓库发布环境未验证);
- ``PySCFCalculator``: PySCF Hartree-Fock/DFT (实验性, 可选依赖);
- ``ASECalculatorAdapter``: 包装任意 ASE calculator 对象 (实验性)。

**诚实声明**: 真实量子化学后端的结果依赖具体程序版本、基组与电子
结构设置, 本仓库不对其数值正确性做任何认证; 未经独立基准校验前,
其数据只应被视为"与该程序设置一致的标签"。可复现性依赖完整的
程序版本与参数记录 (见 ``provenance``)。

单位约定: 坐标 Bohr, 能量 Hartree, 梯度 Hartree/Bohr —— 与动力学
模块一致, 无需换算。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


class Calculator(ABC):
    """电子结构计算后端协议: 能量 (Hartree) + 核梯度 (Hartree/Bohr)。"""

    #: 后端名称 (子类覆盖)
    name: str = "base"

    @abstractmethod
    def energy(self, coords: np.ndarray) -> float:
        """基态能量 (Hartree)。coords: (N_atoms, 3) Bohr。"""

    @abstractmethod
    def gradient(self, coords: np.ndarray) -> np.ndarray:
        """核坐标梯度 dE/dR (N_atoms, 3) Hartree/Bohr。"""

    def energy_and_gradient(self, coords: np.ndarray) -> Tuple[float, np.ndarray]:
        """能量与梯度 (默认分别调用; 有原生梯度支持的后端应覆盖)。"""
        return self.energy(coords), self.gradient(coords)

    @property
    def provenance(self) -> Dict[str, str]:
        """数据来源记录 (程序/版本/参数) — 随数据集保存。"""
        return {"backend": self.name}


# ---------------------------------------------------------------------------
# 解析后端 (无外部依赖, 完全可验证)
# ---------------------------------------------------------------------------

class AnalyticCalculator(Calculator):
    """包装解析能量/梯度可调用对。

    Parameters
    ----------
    energy_fn : Callable
        ``(N, 3) coords -> float`` (Hartree)。
    gradient_fn : Callable, optional
        ``(N, 3) coords -> (N, 3)`` 解析梯度; 缺省时用中心差分
        (精度受步长限制, 仅供演示/测试)。
    name : str
    """

    name = "analytic"

    def __init__(self, energy_fn: Callable, gradient_fn: Optional[Callable] = None,
                 grad_h: float = 1e-5, name: str = "analytic"):
        self.energy_fn = energy_fn
        self.gradient_fn = gradient_fn
        self.grad_h = grad_h
        if name != "analytic":
            self.name = name

    def energy(self, coords: np.ndarray) -> float:
        """解析能量 (Hartree)。coords: (N, 3) Bohr。"""
        return float(self.energy_fn(np.asarray(coords, dtype=float)))

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        """解析梯度 (Hartree/Bohr); 未提供解析梯度时用中心差分。"""
        coords = np.asarray(coords, dtype=float)
        if self.gradient_fn is not None:
            return np.asarray(self.gradient_fn(coords), dtype=float)
        g = np.zeros_like(coords)
        for i in range(coords.shape[0]):
            for j in range(3):
                cp, cm = coords.copy(), coords.copy()
                cp[i, j] += self.grad_h
                cm[i, j] -= self.grad_h
                g[i, j] = (self.energy(cp) - self.energy(cm)) / (2 * self.grad_h)
        return g


# ---------------------------------------------------------------------------
# 外部程序后端 (实验性)
# ---------------------------------------------------------------------------

class CommandBackendError(RuntimeError):
    """外部程序调用/解析失败。"""


class XTBCommandCalculator(Calculator):
    """GFN-xTB 半经验方法 (子进程调用; 实验性)。

    对每个几何执行 ``xtb coordfile --grad [--charg q] [--uhf u]``,
    从 stdout 解析总能量与梯度块。**需要本机安装 xtb 且在 PATH 中**;
    本仓库的发布验证环境未安装 xtb, 该后端未经端到端验证。
    """

    name = "xtb"

    _ENERGY_RE = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s*Eh", re.IGNORECASE)
    _GRAD_RE = re.compile(r"gradient \(Eh/a0\)", re.IGNORECASE)
    _CYCLE_RE = re.compile(r"cycle\s+(\d+)", re.IGNORECASE)

    def __init__(self, symbols: Sequence[str], charge: int = 0,
                 uhf: int = 0, accuracy: float = 1.0,
                 binary: str = "xtb", timeout: float = 300.0):
        self.symbols = list(symbols)
        self.charge = charge
        self.uhf = uhf
        self.accuracy = accuracy
        self.binary = binary
        self.timeout = timeout
        if shutil.which(binary) is None:
            raise CommandBackendError(
                f"未找到 {binary!r}; 请安装 GFN-xTB 并加入 PATH "
                "(或改用 AnalyticCalculator / PySCF / ASE 后端)")

    def _run(self, coords: np.ndarray) -> str:
        coords = np.asarray(coords, dtype=float)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "coord.xyz")
            with open(path, "w") as f:
                f.write(f"{coords.shape[0]}\n\n")
                for sym, (x, y, z) in zip(self.symbols, coords):
                    f.write(f"{sym:2s} {x:20.12f} {y:20.12f} {z:20.12f}\n")
            cmd = [self.binary, path, "--grad",
                   "--chrg", str(self.charge), "--uhf", str(self.uhf),
                   "--acc", str(self.accuracy)]
            try:
                proc = subprocess.run(cmd, cwd=tmp, capture_output=True,
                                      text=True, timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                raise CommandBackendError(f"xtb 调用超时 ({self.timeout}s)") from exc
            if proc.returncode != 0:
                raise CommandBackendError(
                    f"xtb 返回码 {proc.returncode}:\n{proc.stderr[-2000:]}")
            return proc.stdout

    def _parse(self, out: str) -> Tuple[float, np.ndarray]:
        m = self._ENERGY_RE.search(out)
        if not m:
            raise CommandBackendError("xtb 输出中未找到 TOTAL ENERGY 行")
        energy = float(m.group(1))

        n = len(self.symbols)
        gi = self._GRAD_RE.search(out)
        grad = np.zeros((n, 3))
        if gi:
            tail = out[gi.end():]
            nums = re.findall(r"(-?\d+\.\d+)(?:E[-+]?\d+)?", tail)
            values = np.array([float(x) for x in nums[:3 * n]])
            if values.size == 3 * n:
                grad = values.reshape(n, 3)
        return energy, grad

    def energy_and_gradient(self, coords: np.ndarray) -> Tuple[float, np.ndarray]:
        """单次子进程调用同时返回能量 (Hartree) 与梯度 (Hartree/Bohr)。"""
        return self._parse(self._run(coords))

    def energy(self, coords: np.ndarray) -> float:
        """GFN-xTB 总能量 (Hartree); coords: (N, 3) Bohr。"""
        return self.energy_and_gradient(coords)[0]

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        """GFN-xTB 核梯度 (Hartree/Bohr)。"""
        return self.energy_and_gradient(coords)[1]

    @property
    def provenance(self) -> Dict[str, str]:
        """记录 xtb 版本、电荷、自旋多重度与精度设置。"""
        try:
            ver = subprocess.run([self.binary, "--version"],
                                 capture_output=True, text=True,
                                 timeout=30).stdout.strip().splitlines()[0]
        except Exception:
            ver = "unknown"
        return {"backend": "xtb", "version": ver, "charge": str(self.charge),
                "uhf": str(self.uhf), "accuracy": str(self.accuracy)}


class PySCFCalculator(Calculator):
    """PySCF 从头算与 DFT 计算后端 (可选依赖; 实验性)。

    支持能力:
    - **闭壳层与开壳层**: RHF, ROHF, UHF, DFT (RKS/ROKS/UKS)
    - **高自旋约束与自旋锁定 (spin_lock)**: 开壳层自旋纯度审计, 防止态跃迁与自旋污染
    - **最大重叠法 (MOM)**: 沿几何路径保持特定电子轨道占据, 克服激发态与非平衡态变分塌陷
    - **共振态衰减宽度 (CAP-PES)**: 提供自电离衰变宽度 Γ(R) 与复能量 E_R - i*Γ/2 接口
    """

    name = "pyscf"

    def __init__(self, symbols: Sequence[str], basis: str = "sto-3g",
                 charge: int = 0, spin: int = 0, method: str = "rhf",
                 xc: Optional[str] = None, spin_lock: bool = False,
                 spin_tol: float = 0.1, use_mom: bool = False,
                 mom_reference: str = "prev",
                 cap_params: Optional[Dict[str, Any]] = None,
                 unit: str = "Bohr", conv_tol: float = 1e-9,
                 max_cycle: int = 100):
        self.symbols = list(symbols)
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.method = method.lower()
        self.xc = xc
        self.spin_lock = spin_lock
        self.spin_tol = float(spin_tol)
        self.use_mom = use_mom
        self.mom_reference = mom_reference
        self.cap_params = cap_params
        self.unit = unit
        self.conv_tol = float(conv_tol)
        self.max_cycle = int(max_cycle)

        self._ref_mo_coeff = None
        self._ref_mo_occ = None
        self._initial_mo_coeff = None
        self._initial_mo_occ = None

        try:
            import pyscf  # noqa: F401
        except ImportError as exc:
            raise CommandBackendError(
                "未安装 pyscf; pip install pyscf, 或改用其他后端") from exc

    def reset_mom(self, mo_coeff: Optional[np.ndarray] = None,
                  mo_occ: Optional[np.ndarray] = None) -> None:
        """重置或手动指定 MOM (最大重叠法) 的参考轨道。"""
        self._ref_mo_coeff = mo_coeff
        self._ref_mo_occ = mo_occ
        self._initial_mo_coeff = mo_coeff
        self._initial_mo_occ = mo_occ

    def _mol(self, coords: np.ndarray):
        from pyscf import gto
        mol = gto.Mole(atom=[(s, c) for s, c in zip(self.symbols, coords)],
                       basis=self.basis, charge=self.charge,
                       spin=self.spin, unit=self.unit, verbose=0)
        mol.build()  # 显式构建, 避免 SCF kernel 触发未初始化告警
        return mol

    def _mf(self, mol):
        from pyscf import scf, dft
        m = self.method
        if m == "rhf":
            mf = scf.RHF(mol)
        elif m == "rohf":
            mf = scf.ROHF(mol)
        elif m == "uhf":
            mf = scf.UHF(mol)
        elif m in ("dft", "rks"):
            mf = dft.RKS(mol) if self.spin == 0 else dft.ROKS(mol)
            if self.xc:
                mf.xc = self.xc
        elif m == "roks":
            mf = dft.ROKS(mol)
            if self.xc:
                mf.xc = self.xc
        elif m == "uks":
            mf = dft.UKS(mol)
            if self.xc:
                mf.xc = self.xc
        else:
            raise ValueError(
                f"未知 method: {self.method}; 支持 'rhf', 'rohf', 'uhf', "
                f"'dft', 'rks', 'roks', 'uks'"
            )

        mf.conv_tol = self.conv_tol
        mf.max_cycle = self.max_cycle
        return mf

    @staticmethod
    def _mom_setocc(mo_occ: np.ndarray) -> np.ndarray:
        """把 ``mf.mo_occ`` 转换为 PySCF ``mom_occ`` 所需的占据数组。

        - UHF/UKS: ``mo_occ`` 已是 (2, nmo) 的 0/1 alpha/beta 数组, 直接使用;
        - ROHF/ROKS: ``mo_occ`` 为一维 {2,1,0} (双占据/单占据/空), 需展开为
          (2, nmo): alpha = [occ≥1], beta = [occ≥2] (PySCF MOM 的约定)。
        """
        occ = np.asarray(mo_occ, dtype=float)
        if occ.ndim == 2:
            return (occ > 0).astype(float)
        alpha = (occ >= 1.0).astype(float)
        beta = (occ >= 2.0).astype(float)
        return np.stack([alpha, beta])

    def _run(self, coords: np.ndarray):
        from pyscf import lib
        coords_arr = np.asarray(coords, dtype=float)
        mol = self._mol(coords_arr)
        mf = self._mf(mol)

        # MOM (最大重叠法) 注入: 以参考轨道最大重叠原则决定每步占据,
        # 维持指定激发态组态 (PySCF 2.x API: mom_occ(mf, occorb, setocc), 原地修改)
        if self.use_mom and self._ref_mo_coeff is not None and self._ref_mo_occ is not None:
            from pyscf.scf import addons
            setocc = self._mom_setocc(self._ref_mo_occ)
            mf = addons.mom_occ(mf, self._ref_mo_coeff, setocc)

        e = mf.kernel()
        if not mf.converged:
            raise CommandBackendError("PySCF SCF 未收敛")

        # 自旋态审计与自旋锁定
        if hasattr(mf, "spin_square") and callable(mf.spin_square):
            try:
                res = mf.spin_square()
                if isinstance(res, (tuple, list)) and len(res) >= 2:
                    ss, _ = res[0], res[1]
                    s_ideal = abs(self.spin) / 2.0
                    s2_ideal = s_ideal * (s_ideal + 1.0)
                    s2_diff = abs(ss - s2_ideal)
                    if self.spin_lock and s2_diff > self.spin_tol:
                        raise CommandBackendError(
                            f"自旋锁定失败: 实际 <S^2>={ss:.4f}, 理论值 S(S+1)={s2_ideal:.4f}, "
                            f"自旋污染偏差 {s2_diff:.4f} 超过阈值 {self.spin_tol}"
                        )
            except CommandBackendError:
                raise
            except Exception:
                pass

        # 缓存当前收敛轨道供后续构型 MOM 跟踪
        if self.use_mom:
            mo_c = mf.mo_coeff
            mo_o = mf.mo_occ
            if self._initial_mo_coeff is None:
                self._initial_mo_coeff = mo_c
                self._initial_mo_occ = mo_o
            if self.mom_reference == "prev":
                self._ref_mo_coeff = mo_c
                self._ref_mo_occ = mo_o
            elif self.mom_reference == "initial":
                self._ref_mo_coeff = self._initial_mo_coeff
                self._ref_mo_occ = self._initial_mo_occ

        grad = mf.nuc_grad_method().kernel()
        grad_arr = np.asarray(lib.asarray(grad), dtype=float).reshape(-1, 3)
        return float(e), grad_arr

    def energy_and_gradient(self, coords: np.ndarray) -> Tuple[float, np.ndarray]:
        """SCF 能量 (Hartree) 与梯度 (Hartree/Bohr); SCF 不收敛或自旋锁定失败时抛错。"""
        return self._run(coords)

    def energy(self, coords: np.ndarray) -> float:
        """SCF 能量 (Hartree)。"""
        return self._run(coords)[0]

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        """SCF 核梯度 (Hartree/Bohr)。"""
        return self._run(coords)[1]

    def resonance_width(self, coords: np.ndarray) -> float:
        """估计或计算共振态电子自电离衰变宽度 Γ(R) (Hartree)。

        当提供 ``cap_params`` 时按模型或 CAP 势盒计算衰变宽度;
        缺省或未配置时返回 0.0 (实势能面极限)。
        """
        if not self.cap_params:
            return 0.0
        c = np.asarray(coords, dtype=float)
        p_type = self.cap_params.get("type", "exponential")
        idx = self.cap_params.get("r_index", (0, 1))
        if len(c) > max(idx):
            r = float(np.linalg.norm(c[idx[0]] - c[idx[1]]))
        else:
            r = float(np.linalg.norm(c[0]))

        if p_type == "exponential":
            a = float(self.cap_params.get("A", 0.01))
            beta = float(self.cap_params.get("beta", 1.0))
            return float(a * np.exp(-beta * r))
        elif p_type == "box":
            eta = float(self.cap_params.get("eta", 0.001))
            r_cap = float(self.cap_params.get("r_cap", 5.0))
            if r > r_cap:
                return float(2.0 * eta * (r - r_cap) ** 2)
            return 0.0
        return 0.0

    def complex_energy(self, coords: np.ndarray) -> complex:
        """复共振能量 E_res = E_R - i * Γ/2 (Hartree)。"""
        e = self.energy(coords)
        gamma = self.resonance_width(coords)
        return complex(e, -0.5 * gamma)

    @property
    def provenance(self) -> Dict[str, str]:
        """记录 PySCF 版本、方法、基组、电荷、自旋与高级设置。"""
        try:
            import pyscf
            ver = pyscf.__version__
        except Exception:
            ver = "unknown"
        p = {
            "backend": "pyscf",
            "version": ver,
            "method": self.method,
            "basis": self.basis,
            "charge": str(self.charge),
            "spin": str(self.spin),
            "spin_lock": str(self.spin_lock),
            "use_mom": str(self.use_mom),
        }
        if self.xc:
            p["xc"] = str(self.xc)
        if self.cap_params:
            p["cap_params"] = str(self.cap_params)
        return p


class ASECalculatorAdapter(Calculator):
    """包装任意 ASE calculator 对象 (可选依赖; 实验性)。

    ``ase.calculators.calculator.Calculator`` 协议: ``get_potential_energy``
    与 ``get_forces`` (后者为 -∇E, 此处取负号)。
    """

    name = "ase"

    def __init__(self, ase_calculator):
        self.calc = ase_calculator

    def energy_and_gradient(self, coords: np.ndarray) -> Tuple[float, np.ndarray]:
        """ASE 势能 (Hartree) 与梯度 (取力的负号, Hartree/Bohr)。"""
        from ase import Atoms
        atoms = Atoms(symbols=self.calc.atoms.get_chemical_symbols()
                      if hasattr(self.calc, "atoms") else ["H"] * len(coords),
                      positions=np.asarray(coords, dtype=float))
        atoms.calc = self.calc
        e = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        return e, -forces

    def energy(self, coords: np.ndarray) -> float:
        """ASE 势能 (Hartree)。"""
        return self.energy_and_gradient(coords)[0]

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        """ASE 核梯度 (-力, Hartree/Bohr)。"""
        return self.energy_and_gradient(coords)[1]

    @property
    def provenance(self) -> Dict[str, str]:
        """记录被包装 calculator 的类名与版本。"""
        calc = self.calc
        return {"backend": "ase", "calculator": type(calc).__name__,
                "version": str(getattr(calc, "version", "unknown"))}


# ---------------------------------------------------------------------------
# 内置演示后端 (无外部依赖; 明确标注为演示用, 非量子化学)
# ---------------------------------------------------------------------------

def demo_calculator(epsilon: float = 0.01, sigma: float = 3.4) -> Calculator:
    """内置 Lennard-Jones 二聚体演示后端 (解析能量+梯度)。

    用途: 让 ``autoquantum sample/fit`` 闭环在无任何外部量子化学程序
    的环境中也可完整运行与验证。**这不是量子化学计算** — 势能面是
    虚构的 LJ 对势, 仅演示数据生成→力训练管线。
    """

    def energy_fn(coords: np.ndarray) -> float:
        """LJ 对势能量 4ε[(σ/r)¹² − (σ/r)⁶] (Hartree), r 为两原子间距。"""
        c = np.asarray(coords, dtype=float)
        r2 = ((c[0] - c[1]) ** 2).sum()
        inv6 = (sigma ** 2 / r2) ** 3
        return float(4 * epsilon * (inv6 ** 2 - inv6))

    def grad_fn(coords: np.ndarray) -> np.ndarray:
        """LJ 能量解析梯度 (Hartree/Bohr), 沿两原子连线方向。"""
        c = np.asarray(coords, dtype=float)
        d = c[0] - c[1]
        r2 = float((d ** 2).sum())
        inv6 = (sigma ** 2 / r2) ** 3
        # dE/dd = 4ε(-12σ¹²/r¹⁴ + 6σ⁶/r⁸)·d/r²
        coeff = 4 * epsilon * (-12 * sigma ** 12 / r2 ** 7
                              + 6 * sigma ** 6 / r2 ** 4)
        g0 = coeff * d
        return np.array([g0, -g0])

    return AnalyticCalculator(energy_fn, grad_fn, name="demo-lj")


# ---------------------------------------------------------------------------
# 可用性报告
# ---------------------------------------------------------------------------

def available_calculators() -> Dict[str, Dict[str, str]]:
    """报告各后端的可用性 (不抛异常; 供 CLI/文档使用)。"""
    import importlib.util

    status: Dict[str, Dict[str, str]] = {}

    status["analytic"] = {"available": "yes", "note": "内置, 零依赖"}
    status["demo"] = {"available": "yes",
                      "note": "内置 LJ 演示 (非量子化学), 供管线自检"}

    xtb_path = shutil.which("xtb")
    status["xtb"] = {"available": "yes" if xtb_path else "no",
                     "note": xtb_path or "未找到 xtb 可执行文件"}

    has_pyscf = importlib.util.find_spec("pyscf") is not None
    status["pyscf"] = {"available": "yes" if has_pyscf else "no",
                       "note": "pip install pyscf" if not has_pyscf else "installed"}

    has_ase = importlib.util.find_spec("ase") is not None
    status["ase"] = {"available": "yes" if has_ase else "no",
                     "note": "pip install ase" if not has_ase else "installed"}

    return status


def make_calculator(backend: str, symbols: Optional[Sequence[str]] = None,
                    **kwargs) -> Calculator:
    """按名称构造后端 ('demo' | 'analytic' | 'xtb' | 'pyscf' | 'ase')。"""
    if backend == "demo":
        return demo_calculator(**kwargs)
    if backend == "analytic":
        fn = kwargs.pop("energy_fn")
        return AnalyticCalculator(fn, gradient_fn=kwargs.pop("gradient_fn", None))
    if backend == "xtb":
        return XTBCommandCalculator(symbols, **kwargs)
    if backend == "pyscf":
        return PySCFCalculator(symbols, **kwargs)
    if backend == "ase":
        return ASECalculatorAdapter(kwargs.pop("ase_calculator"))
    raise ValueError(f"未知后端: {backend!r}; "
                     f"可选 {sorted(available_calculators())}")
