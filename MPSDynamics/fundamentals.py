import cupy as cp
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

def crea(d: int) -> cp.ndarray:
    """Creation operator for a d-level system."""
    return cp.diag(cp.sqrt(cp.arange(1, d, dtype=cp.complex128)), k=-1)

def anih(d: int) -> cp.ndarray:
    """
    Annihilation operator for d-level system.
    """
    return crea(d).T.conj()

def numb(d: int) -> cp.ndarray:
    """
    Number operator for d-level system.
    """
    return cp.diag(cp.arange(d, dtype=cp.complex128))

def disp(d: int, omega_vib: Optional[float] = None, m: Optional[float] = None) -> cp.ndarray:
    """
    Displacement operator for d-level system.
    """
    return cp.diag(cp.sqrt(cp.arange(1, d, dtype=cp.complex128)), k=1)

def mome(d: int) -> cp.ndarray:
    """
    Momentum operator for d-level system.
    """
    return (1j / cp.sqrt(2)) * (crea(d) - anih(d))

adag_up = cp.diag(cp.array([1, 0, 1], dtype=cp.complex128), k=-1)
adag_dn = cp.diag(cp.array([1, 1], dtype=cp.complex128), k=-2)
a_up = cp.diag(cp.array([1, 0, 1], dtype=cp.complex128), k=1)
a_dn = cp.diag(cp.array([1, 1], dtype=cp.complex128), k=2)
n_tot = cp.diag(cp.array([0, 1, 1, 2], dtype=cp.complex128))
n_up = cp.diag(cp.array([0, 1, 0, 1], dtype=cp.complex128))
n_dn = cp.diag(cp.array([0, 0, 1, 1], dtype=cp.complex128))
parity = cp.diag(cp.array([1, -1, -1, 1], dtype=cp.complex128))

def unitvec(n: int, d: int) -> cp.ndarray:
    """Return the canonical basis vector of dimension d.

        unitvec(0, 3) → [1, 0, 0]
        unitvec(2, 3) → [0, 0, 1]
    """
    if n < 0 or n >= d:
        raise IndexError(f"unitvec: index {n} out of bounds for dimension {d}")
    v = cp.zeros(d, dtype=cp.complex128)
    v[n] = 1.0
    return v

def unitmat(d1: int, d2: Optional[int] = None) -> cp.ndarray:
    """Returns a d1xd2 identity matrix. If d2 is None, returns a d1xd1 matrix."""
    if d2 is None:
        d2 = d1
    return cp.eye(d1, d2, dtype=cp.complex128)

def unitcol(n: int, d: int) -> cp.ndarray:
    """Return the *column* version of unitvec with 0-based index n."""
    if n < 0 or n >= d:
        raise IndexError(f"unitcol: index {n} out of bounds for dimension {d}")
    z = cp.zeros((d, 1), dtype=cp.complex128)
    z[n] = 1.0
    return z

def unitrow(n: int, d: int) -> cp.ndarray:
    """Return the *row* version of unitvec with 0-based index n."""
    if n < 0 or n >= d:
        raise IndexError(f"unitrow: index {n} out of bounds for dimension {d}")
    z = cp.zeros((1, d), dtype=cp.complex128)
    z[0, n] = 1.0
    return z

@dataclass
class SiteOps:
    h: cp.ndarray
    r: cp.ndarray
    l: cp.ndarray

@dataclass
class EnvOps:
    H: cp.ndarray
    op: cp.ndarray
    opd: cp.ndarray

@dataclass
class ObbSite:
    A: cp.ndarray
    V: cp.ndarray

    def __post_init__(self):
        if self.A.shape[1] != self.V.shape[1]:
            raise ValueError("Dimension mismatch")


def eigen_chain(c_params: Tuple[np.ndarray, np.ndarray], num_modes: Optional[int] = None) -> Tuple[cp.ndarray, cp.ndarray]:
    """Constructs and diagonalizes the Hamiltonian for a tight-binding chain."""
    es, ts = c_params
    if num_modes is None:
        num_modes = len(es)
    es = es[:num_modes]
    ts = ts[:num_modes - 1]
    h_mat = cp.diag(es) + cp.diag(ts, k=1) + cp.diag(ts, k=-1)
    eigenvalues, eigenvectors = cp.linalg.eigh(h_mat)
    return eigenvalues, eigenvectors