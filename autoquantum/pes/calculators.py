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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
        return float(self.energy_fn(np.asarray(coords, dtype=float)))

    def gradient(self, coords: np.ndarray) -> np.ndarray:
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
        return self._parse(self._run(coords))

    def energy(self, coords: np.ndarray) -> float:
        return self.energy_and_gradient(coords)[0]

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        return self.energy_and_gradient(coords)[1]

    @property
    def provenance(self) -> Dict[str, str]:
        try:
            ver = subprocess.run([self.binary, "--version"],
                                 capture_output=True, text=True,
                                 timeout=30).stdout.strip().splitlines()[0]
        except Exception:
            ver = "unknown"
        return {"backend": "xtb", "version": ver, "charge": str(self.charge),
                "uhf": str(self.uhf), "accuracy": str(self.accuracy)}


class PySCFCalculator(Calculator):
    """PySCF RHF/DFT 后端 (可选依赖; 实验性)。

    需安装 pyscf; 梯度经 SCF 的 ``grad`` kernel 获得 (RHF 为解析
    梯度, DFT 数值梯度)。本仓库发布验证环境未安装 pyscf。
    """

    name = "pyscf"

    def __init__(self, symbols: Sequence[str], basis: str = "sto-3g",
                 charge: int = 0, spin: int = 0, method: str = "rhf",
                 unit: str = "Bohr"):
        self.symbols = list(symbols)
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.method = method.lower()
        self.unit = unit
        try:
            import pyscf  # noqa: F401
        except ImportError as exc:
            raise CommandBackendError(
                "未安装 pyscf; pip install pyscf, 或改用其他后端") from exc

    def _mol(self, coords: np.ndarray):
        from pyscf import gto
        return gto.Mole(atom=[(s, c) for s, c in zip(self.symbols, coords)],
                        basis=self.basis, charge=self.charge,
                        spin=self.spin, unit=self.unit, verbose=0)

    def _mf(self, mol):
        if self.method == "rhf":
            from pyscf import scf
            return scf.RHF(mol)
        elif self.method in ("uhf", "rohf"):
            from pyscf import scf
            return scf.UHF(mol) if self.method == "uhf" else scf.ROHF(mol)
        elif self.method == "dft":
            from pyscf import dft
            return dft.KS(mol)
        raise ValueError(f"未知 method: {self.method}")

    def _run(self, coords: np.ndarray):
        from pyscf import lib
        mol = self._mol(np.asarray(coords, dtype=float))
        mf = self._mf(mol)
        e = mf.kernel()
        if not mf.converged:
            raise CommandBackendError("PySCF SCF 未收敛")
        grad = mf.nuc_grad_method().kernel()
        return float(e), np.asarray(lib.asarray(grad), dtype=float).reshape(-1, 3)

    def energy_and_gradient(self, coords: np.ndarray) -> Tuple[float, np.ndarray]:
        return self._run(coords)

    def energy(self, coords: np.ndarray) -> float:
        return self._run(coords)[0]

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        return self._run(coords)[1]

    @property
    def provenance(self) -> Dict[str, str]:
        try:
            import pyscf
            ver = pyscf.__version__
        except Exception:
            ver = "unknown"
        return {"backend": "pyscf", "version": ver, "method": self.method,
                "basis": self.basis, "charge": str(self.charge),
                "spin": str(self.spin)}


class ASECalculatorAdapter(Calculator):
    """包装任意 ASE calculator 对象 (可选依赖; 实验性)。

    ``ase.calculators.calculator.Calculator`` 协议: ``get_potential_energy``
    与 ``get_forces`` (后者为 -∇E, 此处取负号)。
    """

    name = "ase"

    def __init__(self, ase_calculator):
        self.calc = ase_calculator

    def energy_and_gradient(self, coords: np.ndarray) -> Tuple[float, np.ndarray]:
        from ase import Atoms
        atoms = Atoms(symbols=self.calc.atoms.get_chemical_symbols()
                      if hasattr(self.calc, "atoms") else ["H"] * len(coords),
                      positions=np.asarray(coords, dtype=float))
        atoms.calc = self.calc
        e = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        return e, -forces

    def energy(self, coords: np.ndarray) -> float:
        return self.energy_and_gradient(coords)[0]

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        return self.energy_and_gradient(coords)[1]

    @property
    def provenance(self) -> Dict[str, str]:
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
        c = np.asarray(coords, dtype=float)
        r2 = ((c[0] - c[1]) ** 2).sum()
        inv6 = (sigma ** 2 / r2) ** 3
        return float(4 * epsilon * (inv6 ** 2 - inv6))

    def grad_fn(coords: np.ndarray) -> np.ndarray:
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
