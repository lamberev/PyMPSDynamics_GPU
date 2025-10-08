import cupy as cp
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Tuple, List, Optional, Any

from .tensorOps import rho_aa_star, rho_aoa_star, is_hermitian

# --- Observable Dataclasses ---

@dataclass
class Observable:
    name: str

@dataclass
class OneSiteObservable(Observable):
    op: cp.ndarray
    sites: Union[int, Tuple[int, int], List[int], None] = None
    hermitian: bool = field(init=False)
    allsites: bool = field(init=False, default=False)

    def __post_init__(self):
        self.hermitian = is_hermitian(self.op)
        if self.sites is None:
            self.allsites = True # Default to measuring on all sites if no sites are specified

    def __eq__(self, other): # Define meaning of "==" in "Obs1 == Obs2"
        if not isinstance(other, OneSiteObservable): # Ensure other object is also a OneSiteObservable
            return NotImplemented
        
        # Note: op comparison is by reference, which is what you generally want for np/cp arrays as they can be large
        # This may be unnecessary
        return (self.name == other.name and 
                self.op is other.op and  # Compare by reference (== checks by value for np/cp arrays which is slow); checking by reference is fast and amounts to a pointer comparison to see if the two arrays are the same object in memory
                self.sites == other.sites)

    # This may be unnecessary
    def __hash__(self):
        # The operator `op` is a numpy/cupy array, which is not hashable.
        # We can use its id() for the hash, but it's better to rely on attributes
        # that define the observable's identity. `name` should be unique.
        # To handle mutable list `sites`, we convert to tuple.
        
        sites_tuple = None
        if isinstance(self.sites, list):
            sites_tuple = tuple(self.sites)
        elif isinstance(self.sites, (int, tuple)):
            sites_tuple = self.sites

        return hash((self.name, sites_tuple))

@dataclass
class TwoSiteObservable(Observable):
    op1: cp.ndarray
    op2: cp.ndarray
    sites1: Union[int, Tuple[int, int], List[int], None] = None
    sites2: Union[int, Tuple[int, int], List[int], None] = None
    allsites: bool = field(init=False, default=False)

    def __post_init__(self):
        if self.sites1 is None and self.sites2 is None:
            self.allsites = True

@dataclass
class RhoReduced(Observable):
    sites: Union[int, Tuple[int, int]]

@dataclass(init=False)
class CdagCup(Observable):
    sites: Tuple[int, int]
    
    def __init__(self, sites: Tuple[int, int], name: str = "CdagCup"):
        super().__init__(name)
        self.sites = sites

@dataclass(init=False)
class CdagCdn(Observable):
    sites: Tuple[int, int]

    def __init__(self, sites: Tuple[int, int], name: str = "CdagCdn"):
        super().__init__(name)
        self.sites = sites
    
def _span_sites(A: List[cp.ndarray], sites: Union[int, Tuple[int, int], List[int], None]) -> List[int]:
    """Helper to get a list of sites to measure."""
    if sites is None:
        return list(range(len(A)))

    # The 'sites' argument is 1-based as in the Julia code.
    # We convert to a 0-based list for Python.
    
    if isinstance(sites, int):
        return [sites - 1]
    
    unadjusted_sites: List[int] = []
    if isinstance(sites, tuple):
        start, end = sites
        if start <= end:
            unadjusted_sites = list(range(start, end + 1))
        else:  # Reversed range
            unadjusted_sites = list(range(start, end - 1, -1))
    elif isinstance(sites, list):
        unadjusted_sites = sites
    else:
        # Should be unreachable with current types
        return sites

    return [s - 1 for s in unadjusted_sites] # Check if this is correct


def measure(A: List[cp.ndarray], obs: Any, **kwargs):
    """
    Dispatcher function to measure an observable `obs` on an MPS `A`.
    NOTE: Assumes MPS A is right-normalized.
    """
    # Pre-calculate left environments if needed for efficiency
    if 'rho' not in kwargs:
        max_site = len(A) # Simplified for now
        kwargs['rho'] = left_contract_mps(A, max_site)

    if isinstance(obs, OneSiteObservable):
        return measure_1site_operator(A, obs.op, obs.sites, **kwargs)
    elif isinstance(obs, TwoSiteObservable):
        return measure_2site_operator(A, obs.op1, obs.op2, obs.sites1, obs.sites2, **kwargs)
    elif isinstance(obs, RhoReduced):
        if isinstance(obs.sites, int):
            return rho_reduced_1site(A, obs.sites)
        else:
            return rho_reduced_2sites(A, obs.sites)
    elif isinstance(obs, (CdagCup, CdagCdn)):
        return measure_cdag_c(A, obs, **kwargs)
    elif isinstance(obs, list):
        return [measure(A, o, **kwargs) for o in obs]
    else:
        raise TypeError(f"Measurement for observable of type {type(obs)} is not supported.")

def measure_1site_operator(
    A: List[cp.ndarray], 
    O: cp.ndarray, 
    sites: Union[int, Tuple[int, int], List[int], None] = None,
    rho: Optional[List[cp.ndarray]] = None,
    **kwargs
) -> Union[float, complex, cp.ndarray]:
    """
    Computes the expectation value of a one-site operator O.
    Corresponds to multiple `measure1siteoperator` functions in Julia.
    """
    N = len(A)
    T = cp.float64 if is_hermitian(O) else cp.complex128
    
    if rho is not None:
        # Use pre-computed environments
        site_list = _span_sites(A, sites)
        expvals = []
        for i in site_list:
            v = rho_aoa_star(rho[i], A[i], O, indir=1)
            if T == cp.float64:
                v = v.real
            expvals.append(v)
        
        if isinstance(sites, int):
            return expvals[0]
        if isinstance(sites, tuple) and sites[0] > sites[1]:
            return cp.array(expvals, dtype=T)[::-1]
        return cp.array(expvals, dtype=T)

    # Contract from scratch
    if sites is None:
        # Corresponds to measure1siteoperator(A, O) in Julia
        rho_env = cp.eye(A[0].shape[0], dtype=complex).reshape(1, 1)
        expvals = cp.zeros(N, dtype=T)
        for i in range(N):
            v = rho_aoa_star(rho_env, A[i], O, indir=1)
            if T == cp.float64:
                v = v.real
            expvals[i] = v
            if i < N - 1:
                rho_env = rho_aa_star(rho_env, A[i])
        return expvals
    
    if isinstance(sites, int):
        # Corresponds to measure1siteoperator(A, O, site::Int) in Julia
        rho_env = cp.eye(A[0].shape[0], dtype=complex).reshape(1, 1)
        for i in range(sites):
            rho_env = rho_aa_star(rho_env, A[i])
        v = rho_aoa_star(rho_env, A[sites], O, indir=1)
        return v.real if T == cp.float64 else v

    if isinstance(sites, tuple):
        # Corresponds to measure1siteoperator(A, O, chainsection::Tuple{Int64,Int64}) in Julia
        l, r = min(sites), max(sites)
        rev = sites[0] > sites[1]
        num_sites = r - l + 1
        expvals = cp.zeros(num_sites, dtype=T)
        
        rho_env = cp.eye(A[0].shape[0], dtype=complex).reshape(1, 1)
        for i in range(l):
            rho_env = rho_aa_star(rho_env, A[i])
        
        for i in range(l, r + 1):
            v = rho_aoa_star(rho_env, A[i], O, indir=1)
            expvals[i-l] = v.real if T == cp.float64 else v
            if i < r:
                rho_env = rho_aa_star(rho_env, A[i])
        
        return expvals[::-1] if rev else expvals

    if isinstance(sites, list):
        # Corresponds to measure1siteoperator(A, O, sites::AbstractVector{Int}) in Julia
        # This implementation mirrors the Julia version for a generic list of sites.
        expvals_full = cp.zeros(N, dtype=T)
        rho_env = cp.eye(A[0].shape[0], dtype=complex).reshape(1, 1)
        for i in range(N):
            v = rho_aoa_star(rho_env, A[i], O, indir=1)
            expvals_full[i] = v.real if T == cp.float64 else v
            if i < N - 1:
                rho_env = rho_aa_star(rho_env, A[i])
        
        site_indices = cp.array(sites)
        return expvals_full[site_indices]
        
def measure_2site_operator(
    A: List[cp.ndarray],
    M1: cp.ndarray,
    M2: cp.ndarray,
    sites1: Union[int, Tuple[int, int], List[int], None] = None,
    sites2: Union[int, Tuple[int, int], List[int], None] = None,
    rho: Optional[List[cp.ndarray]] = None,
    **kwargs
) -> Union[float, complex, cp.ndarray]:
    """
    Computes the expectation value of a two-site operator <M1_i M2_j>.
    """
    
    N = len(A)
    herm_trans = is_hermitian(M1) and is_hermitian(M2)
    T = cp.float64 if herm_trans else cp.complex128
    
    # Case: Single pair of sites (j1, j2)
    if isinstance(sites1, int) and isinstance(sites2, int):
        j1, j2 = sites1, sites2
        if j1 == j2:
            op_prod = M1 @ M2
            return measure_1site_operator(A, op_prod, j1, rho=rho)
        
        i1, i2 = min(j1, j2), max(j1, j2)
        m1, m2 = (M1, M2) if j1 < j2 else (M2, M1)
        
        rho_env = rho[i1] if rho else left_contract_mps(A, i1+1)[i1]
        
        rho1 = rho_aoa_star(rho_env, A[i1], m1, indir=1)
        
        for k in range(i1 + 1, i2):
            rho1 = rho_aa_star(rho1, A[k])
        
        v = rho_aoa_star(rho1, A[i2], m2, indir=1)
        return v.real if herm_trans else v

    # Case: All-to-all measurement
    if sites1 is None and sites2 is None:
        cpair = cp.allclose(M1, M2.conj().T)
        if cp.allclose(M1, M2) or cpair:
             return measure_2site_operator_pair(A, M1, sites=None, conjugate=cpair, rho=rho)
        
        herm_cis = is_hermitian(M1 @ M2)
        T_full = cp.float64 if (herm_cis and herm_trans) else cp.complex128
        expval = cp.zeros((N, N), dtype=T_full)
        
        rho_envs = rho if rho else left_contract_mps(A, N)

        for i in range(N):
            rho_i = rho_envs[i]
            # Diagonal part
            v_diag = rho_aoa_star(rho_i, A[i], M1 @ M2, indir=1)
            expval[i, i] = v_diag.real if herm_cis else v_diag
            
            # Off-diagonal part
            rho12 = rho_aoa_star(rho_i, A[i], M1, indir=1)
            rho21 = rho_aoa_star(rho_i, A[i], M2, indir=1)
            for j in range(i + 1, N):
                v12 = rho_aoa_star(rho12, A[j], M2, indir=1)
                expval[i, j] = v12.real if herm_trans else v12
                
                v21 = rho_aoa_star(rho21, A[j], M1, indir=1)
                expval[j, i] = v21.real if herm_trans else v21

                if j < N - 1:
                    rho12 = rho_aa_star(rho12, A[j])
                    rho21 = rho_aa_star(rho21, A[j])
        return expval

    # Case: General lists of sites
    s1_list = _span_sites(A, sites1)
    s2_list = _span_sites(A, sites2)
    
    if M1.shape == M2.shape:
        op_prod = M1 @ M2
        herm_cis = is_hermitian(op_prod)
    else:
        herm_cis = False

    max_site = max(max(s1_list, default=-1), max(s2_list, default=-1))
    expval = cp.zeros((N, N), dtype=T)
    
    rho_envs = rho if rho else left_contract_mps(A, max_site + 1)

    for i in range(max_site + 1):
        rho_i = rho_envs[i]
        is_in_s1 = i in s1_list
        is_in_s2 = i in s2_list

        if is_in_s1 and is_in_s2 and M1.shape == M2.shape:
            v = rho_aoa_star(rho_i, A[i], op_prod, indir=1)
            expval[i, i] = v.real if herm_cis else v
        
        if is_in_s1:
            rho12 = rho_aoa_star(rho_i, A[i], M1, indir=1)
            for j in range(i + 1, max_site + 1):
                if j in s2_list:
                    v = rho_aoa_star(rho12, A[j], M2, indir=1)
                    expval[i, j] = v.real if herm_trans else v
                if j < max_site:
                    rho12 = rho_aa_star(rho12, A[j])

        if is_in_s2:
            rho21 = rho_aoa_star(rho_i, A[i], M2, indir=1)
            for j in range(i + 1, max_site + 1):
                if j in s1_list:
                    v = rho_aoa_star(rho21, A[j], M1, indir=1)
                    expval[j, i] = v.real if herm_trans else v
                if j < max_site:
                    rho21 = rho_aa_star(rho21, A[j])
                    
    s1_indices = np.array(s1_list)
    s2_indices = np.array(s2_list)
    return expval[s1_indices[:, None], s2_indices]

def measure_2site_operator_pair(
    A: List[cp.ndarray],
    M1: cp.ndarray,
    sites: Optional[Union[List[int], Tuple[int, int]]] = None,
    conjugate: bool = False,
    rho: Optional[List[cp.ndarray]] = None
) -> cp.ndarray:
    """Optimized version for operators like <O_i O_j> or <O_i O^dagger_j>."""
    from .tensorOps import rho_aa_star, rho_aoa_star, is_hermitian
    N = len(A)
    
    if isinstance(sites, tuple):
        site_list = list(range(sites[0], sites[1] + 1))
        rev = sites[0] > sites[1]
        if rev:
            site_list = list(range(sites[0], sites[1] - 1, -1))
    else:
        site_list = sites if sites is not None else list(range(N))
        rev = False

    site_set = set(site_list)
    M2 = M1.conj().T if conjugate else M1
    
    op_prod = M1 @ M2
    herm_cis = is_hermitian(op_prod)
    herm_trans = is_hermitian(M1) and is_hermitian(M2)
    T = cp.float64 if (herm_cis and herm_trans) else cp.complex128
    
    expval_full = cp.zeros((N, N), dtype=T)
    
    rho_env = cp.eye(A[0].shape[0], dtype=cp.complex128).reshape(1, 1)

    for i in range(N):
        rho_i = rho[i] if rho is not None else rho_env
        if i in site_set:
            # Diagonal
            v_diag = rho_aoa_star(rho_i, A[i], op_prod, indir=1)
            expval_full[i, i] = v_diag.real if herm_cis else v_diag
            
            # Off-diagonal
            rho12 = rho_aoa_star(rho_i, A[i], M1, indir=1)
            for j in range(i + 1, N):
                if j in site_set:
                    v = rho_aoa_star(rho12, A[j], M2, indir=1)
                    expval_full[i, j] = v.real if herm_trans else v
                if j < N:
                    rho12 = rho_aa_star(rho12, A[j])
        
        if rho is None and i < N - 1:
            rho_env = rho_aa_star(rho_env, A[i])
            
    # Sub-select the required sites
    indices = np.array(site_list)
    expval = expval_full[np.ix_(indices, indices)]

    dia = cp.diag(cp.diag(expval))
    expval = expval + (expval.conj().T if conjugate else expval.T) - dia
    
    if rev:
        expval = cp.flip(expval, axis=(0,1))
        
    return expval


def measure_cdag_c(A: List[cp.ndarray], obs: Union[CdagCup, CdagCdn], rho: Optional[List[cp.ndarray]] = None, **kwargs):
    """Measures <c_i^dagger c_j> correlations."""
    from .tensorOps import rho_aoa_star
    from .fundamentals import adag_up, a_up, parity, adag_dn, a_dn
    first, last = obs.sites
    nearest, farthest = min(first, last), max(first, last)
    N_sites = farthest - nearest + 1
    expval = cp.zeros((N_sites, N_sites), dtype=cp.complex128)

    # Determine operators based on observable type
    if isinstance(obs, CdagCup):
        op_diag = adag_up @ a_up
        op_prop_left = adag_up @ parity
        op_prop_right = a_up
        op_string = parity
    else: # CdagCdn
        op_diag = adag_dn @ a_dn
        op_prop_left = adag_dn
        op_prop_right = a_dn
        op_string = parity

    if rho is None:
        rho_env = cp.eye(A[0].shape[0], dtype=cp.complex128).reshape(1, 1)
        for i_offset, i in enumerate(range(nearest, farthest + 1)):
            if i > 0:
                rho_env = rho_aa_star(rho_env, A[i-1])

            # Diagonal part
            expval[i_offset, i_offset] = rho_aoa_star(rho_env, A[i], op_diag, indir=1)

            # Off-diagonal part
            rho1 = rho_aoa_star(rho_env, A[i], op_prop_left, indir=1)
            for j_offset, j in enumerate(range(i + 1, farthest + 1)):
                v = rho_aoa_star(rho1, A[j], op_prop_right, indir=1)
                expval[i_offset, i_offset + j_offset + 1] = v
                expval[i_offset + j_offset + 1, i_offset] = v.conj()

                if j < farthest:
                    rho1 = rho_aa_star(rho_aa_star(rho1, A[j-1]), A[j], op_string, indir=1)
    else:
        for i_offset, i in enumerate(range(nearest, farthest + 1)):
            rho_i = rho[i]
            
            # Diagonal part
            expval[i_offset, i_offset] = rho_aoa_star(rho_i, A[i], op_diag, indir=1)
            
            # Off-diagonal part
            rho1 = rho_aoa_star(rho_i, A[i], op_prop_left, indir=1)
            for j_offset, j in enumerate(range(i + 1, farthest + 1)):
                v = rho_aoa_star(rho1, A[j], op_prop_right, indir=1)
                expval[i_offset, i_offset + j_offset + 1] = v
                expval[i_offset + j_offset + 1, i_offset] = v.conj()
                
                if j < farthest:
                    rho1 = rho_aa_star(rho_aa_star(rho_i, A[j-1]), A[j], op_string, indir=1)
    
    if nearest != first: # Reverse if sites were (last, first)
        expval = cp.flip(expval, axis=(0,1))
    return expval

def left_contract_mps(A: List[cp.ndarray], N_sites: int) -> List[cp.ndarray]:
    """Pre-calculates left environments up to N_sites."""
    from .tensorOps import rho_aa_star
    if N_sites < 0:
        return []
    # rhos[i] will be the left environment of site i (0-indexed)
    # So rhos will have N_sites+1 elements, from 0 to N_sites
    rhos = [None] * (N_sites + 1)
    rhos[0] = cp.eye(A[0].shape[0], dtype=complex).reshape(1, 1) # bond dim of first site
    for i in range(N_sites):
        # Skip internal nodes (rank > 3) when building chain-style environments.
        # Julia's `leftcontractmps` is defined only for conventional chain MPS
        # where each tensor has at most one *child* bond. In a tree-MPS these
        # internal nodes would break that assumption and must therefore be
        # ignored when we want a simple left-to-right sweep.
        if A[i].ndim <= 3:
            rhos[i+1] = rho_aa_star(rhos[i], A[i])
        else:
            # Keep the environment unchanged – it will be updated when we hit
            # the actual chain (leaf) tensors that follow this internal node.
            rhos[i+1] = rhos[i]
    rho_list = rhos[:N_sites+1]
    rho_list[N_sites-1] = rhos[N_sites]
    return rho_list

def rho_reduced_1site(A: List[cp.ndarray], site: int = 0) -> cp.ndarray:
    """
    Calculates the reduced density matrix of the MPS A at the specified site.
    Assumes the MPS is right-normalized.
    """
    from .tensorOps import rho_aa_star 
    N = len(A)
    site_idx = site
    if site < 0:
        site_idx += N

    # Contract right environment
    rho_R = cp.eye(A[-1].shape[1], dtype=complex).reshape(1, 1)
    for i in range(N - 1, site_idx, -1):
        rho_R = rho_aa_star(rho_R, A[i], contract_from='right')

    # Contract left environment
    rho_L = cp.eye(A[0].shape[0], dtype=complex).reshape(1, 1)
    for i in range(site_idx):
        rho_L = rho_aa_star(rho_L, A[i])

    # Final contraction at the site
    # A is (left_bond, right_bond, phys) -> (a, c, e)
    # A_conj is (left_bond_conj, right_bond_conj, phys_conj) -> (b, d, f)
    # rho_L is (left_bond_conj, left_bond) -> (b, a)
    # rho_R is (right_bond_conj, right_bond) -> (d, c)
    # Contraction: rho_L[b,a] * A[a,c,e] * A_conj[b,d,f] * rho_R[d,c] -> [e,f]
    temp = cp.einsum('dc,ace->ade', rho_R, A[site_idx], optimize=True)
    rho_reduced = cp.einsum('ba,ade,bdf->ef', rho_L, temp, A[site_idx].conj(), optimize=True)
    return rho_reduced

def rho_reduced_2sites(A: List[cp.ndarray], sites: Tuple[int, int]) -> cp.ndarray:
    """
    Calculates the reduced density matrix of two sites of the MPS A.
    Assumes the MPS is right-normalized.
    """
    from .tensorOps import rho_aa_star
    N = len(A)
    site1, site2 = sites
    
    if site2 != site1 + 1:
        raise NotImplementedError("Reduced density matrix for non-neighboring sites is not implemented.")

    # Contract right environment up to site2
    rho_R = cp.eye(A[-1].shape[1], dtype=complex).reshape(1, 1)
    for i in range(N - 1, site2, -1):
        rho_R = rho_aa_star(rho_R, A[i], contract_from='right')

    # Contract left environment up to site1
    rho_L = cp.eye(A[0].shape[0], dtype=complex).reshape(1, 1)
    for i in range(site1):
        rho_L = rho_aa_star(rho_L, A[i])
        
    A1 = A[site1]
    A2 = A[site2]
    # ρreduced1[a,b,s,s'] := ρR[a0,b0] * conj(A[site2][a,a0,s']) * A[site2][b,b0,s]
    rho_reduced1 = cp.einsum('ij,kil,mjn->kmln', rho_R, A2.conj(), A2, optimize=True)
    
    # ρreduced2[a,b,s,s'] := ρL[a0,b0] * conj(A[site1][a0,a,s']) * A[site1][b0,b,s]
    rho_reduced2 = cp.einsum('ij,kim,jln->kmln', rho_L, A1.conj(), A1, optimize=True)

    # ρreduced[s,d1,s',d2] := ρreduced2[a0,b0,d1,d2] * ρreduced1[a0,b0,s,s']
    rho_reduced = cp.einsum('kmln,kmen->lnen', rho_reduced2, rho_reduced1, optimize=True) # These should be combined for efficiency possibly
    
    d1 = A1.shape[2]
    d2 = A2.shape[2]
    return rho_reduced.reshape(d1, d2, d1, d2)


def measure_mpo(A: List[cp.ndarray], M: List[cp.ndarray], sites: Optional[Tuple[int, int]] = None) -> float:
    """
    For a list of tensors `A` representing a right orthonormalized MPS, compute the local expectation
    value of the MPO M.
    """
    from .tensorOps import update_left_env

    N = len(M)
    if N != len(A):
        raise ValueError(f"MPO has {N} sites while MPS has {len(A)} sites")

    F = cp.ones((1, 1, 1), dtype=complex)
    
    if sites is None:
        # Measure over all sites
        for k in range(N):
            F = update_left_env(A[k], M[k], F)
        return F.item().real
    else:
        # Measure over specified range of sites
        l, r = sites
        for k in range(l):
            d = A[k].shape[2]
            id_op = cp.eye(d).reshape(1, 1, d, d)
            F = update_left_env(A[k], id_op, F)
        
        for k in range(l, r + 1):
            F = update_left_env(A[k], M[k], F)
        
        # Final contraction to get the expectation value
        # In Julia: F = tensortrace([2], F, [1,2,1])
        # This is a trace over the MPO bond dimension
        val = cp.einsum('aba->', F)
        return val.item().real

# Think this is unnecessary; defined in tensorOps.py
def is_hermitian(op: cp.ndarray, tol: float = 1e-9) -> bool:
    if op.ndim != 2 or op.shape[0] != op.shape[1]:
        return False
    return cp.allclose(op, op.conj().T, atol=tol)