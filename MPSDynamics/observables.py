import cupy as cp
from dataclasses import dataclass
from typing import Any, Tuple, List
import numpy as np

from .fundamentals import eigen_chain, disp, mome, anih, crea, unitmat, unitcol, unitrow
from .mpsBasics import phys_dims
from .measure import measure, OneSiteObservable, TwoSiteObservable, left_contract_mps


@dataclass
class FockError:
    """
    An observable for calculating a specific error metric related to a
    system coupled to a Fock (oscillator) bath.
    """
    name: str
    cparams: List
    sysop: cp.ndarray
    sites: Tuple[int, int]
    eigensys: Any

    def __init__(self, cpars, sysop, sites):
        self.name = "FockError"
        self.cparams = cpars
        self.sysop = sysop
        self.sites = sites
        self.eigensys = eigen_chain(cpars)

    @classmethod
    def from_sysop_and_n(cls, sysop: cp.ndarray, N: int):
        """
        Convenience constructor for FockError.
        """
        cpars = [cp.zeros(N), cp.ones(N - 1), 0.5]
        sites = (1, N + 1)
        return cls(cpars, sysop, sites)
    
# Need to go back and check if this is correct
def measure_fock_error(A: List[cp.ndarray], obs: FockError, t: float = 0, **kwargs) -> cp.ndarray:
    """
    Return the measure of the FockError observable on the MPS A.
    """
    site_slice = slice(obs.sites[0], obs.sites[1])
    pdims = phys_dims(A)[site_slice]
    d = pdims[0]
    if not all(pd == d for pd in pdims):
        raise ValueError("MPS has non-uniform local Hilbert space dimensions in the specified range")

    x = disp(d)
    p = mome(d)
    h = obs.cparams[2] * obs.sysop
    U = obs.eigensys.vectors
    S = obs.eigensys.values
    N = len(S)
    
    # In Julia, U' is the adjoint. For a unitary matrix from eigen(), U' is U.conj().T
    # ca = (U' * diagm(...) * U).T
    ca = (U.conj().T @ cp.diag(cp.exp(1j * t * S)) @ U).T

    # To simplify mapping from Julia, we construct `c` so that c_py[r-1, s-1] corresponds to c_jl[r,s].
    # Julia's c = reshape([real(ca), -imag(ca), imag(ca), real(ca)], 2, 2) is column-major.
    # c_jl[1,1]=real(ca), c_jl[2,1]=-imag(ca), c_jl[1,2]=imag(ca), c_jl[2,2]=real(ca)
    c = cp.array([[ca.real, ca.imag], [-ca.imag, ca.real]]) # c is a (2, 2, N, N) tensor

    gamma = cp.zeros((2, 2, N, N), dtype=ca.dtype)
    uc1 = unitcol(0, N, dtype=ca.dtype).flatten()
    ur1 = unitrow(0, N, dtype=ca.dtype).flatten()
    
    c11_row1 = c[0, 0, 0, :]  # First row of real(ca)
    c12_row1 = c[0, 1, 0, :]  # First row of imag(ca)

    # Replicating the gamma calculation from Julia
    for r_idx, r in enumerate(range(1, 3)): # r = 1, 2
        for s_idx, s in enumerate(range(1, 3)): # s = 1, 2
            # Corresponds to Julia's c[1,r] and c[2,r]
            c1r, c2r = c[0, r_idx, :, :], c[1, r_idx, :, :]
            c1s, c2s = c[0, s_idx, :, :], c[1, s_idx, :, :]

            vec_r = c11_row1 @ c1r + c12_row1 @ c2r
            vec_s = c11_row1 @ c1s + c12_row1 @ c2s

            term1 = cp.outer(vec_r.conj(), vec_s)
            term2 = cp.outer(uc1, vec_s) if r == 1 else cp.zeros_like(term1)
            term3 = cp.outer(vec_r.conj(), ur1) if s == 1 else cp.zeros_like(term1)
            term4 = cp.outer(uc1, ur1) if s == 1 and r == 1 else cp.zeros_like(term1)
            gamma[r_idx, s_idx, :, :] = term1 - term2 - term3 + term4

    rho = left_contract_mps(A, [h**2])

    u = unitmat(d + 1)
    ud_diag = cp.array([1]*d + [0])
    ud = cp.diag(ud_diag)
    o_full = ud @ anih(d + 1) @ (u - ud) @ crea(d + 1) @ ud
    o = o_full[:d, :d]
        
    aad = measure(A, OneSiteObservable("", o, obs.sites), rho=rho)
    oxx = aad
    oxp = 1j * aad
    opx = -1j * aad
    opp = aad
    ors = cp.array([[oxx, opx], [oxp, opp]])
    
    h2xx = measure(A, TwoSiteObservable("", x, x, obs.sites), rho=rho)
    h2pp = measure(A, TwoSiteObservable("", p, p, obs.sites), rho=rho)
    h2xp = measure(A, TwoSiteObservable("", x, p, obs.sites), rho=rho)
    h2px = h2xp.T.conj()
    h2rs = cp.array([[h2xx, h2px], [h2xp, h2pp]])

    # result is the sum of the two error terms, eps1 and eps2.
    eps1 = cp.sum(gamma * h2rs)
    
    eps2 = 0.0
    # In Julia: sum([c[1,r][1,k] * c[1,s][1,k] * ors[r,s][k] ... ])
    for r_idx in range(2):
        for s_idx in range(2):
            c1r_row1 = c[0, r_idx, 0, :]
            c1s_row1 = c[0, s_idx, 0, :]
            eps2 += cp.sum(c1r_row1 * c1s_row1 * ors[r_idx, s_idx, :])

    return gamma * h2rs

def errorbar(op: cp.ndarray, e: cp.ndarray, t: cp.ndarray) -> cp.ndarray:
    """
    Calculates an integrated error estimate from a time series.
    """
    Nt = len(t)
    eave = cp.array([(t[i+1]-t[i]) * cp.sqrt((e[i]+e[i+1])/2) for i in range(Nt-1)])
    integral = cp.zeros(Nt, dtype=eave.dtype)
    integral[1:] = cp.cumsum(eave)
    op_norm = cp.linalg.norm(op, ord=2) # Operator 2-norm
    return cp.sqrt(8) * op_norm * cp.sqrt(integral) 