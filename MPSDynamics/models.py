import cupy as cp
import h5py
import numpy as np
from typing import List, Union, Tuple

from .fundamentals import anih, crea, numb, unitmat, unitcol
from .mpsBasics import reverse_mpo
from .treeBasics import TreeNetwork

def up(*ops: cp.ndarray) -> cp.ndarray:
    """
    Stack operators along a new first dimension.
    """
    ops_complex = [op.astype(cp.complex128) for op in ops]
    stacked = cp.stack(ops_complex, axis=2)
    result = cp.transpose(stacked, (2, 0, 1))
    
    return result

def dn(*ops: cp.ndarray) -> cp.ndarray:
    """
    Stack operators along a new first dimension in reverse order.
    """
    reversed_ops = ops[::-1]
    ops_complex = [op.astype(cp.complex128) for op in reversed_ops]
    stacked = cp.stack(ops_complex, axis=2)
    result = cp.transpose(stacked, (2, 0, 1))
    
    return result

def h_bath_chain(
    N: int,
    d: int,
    chainparams: Tuple[list, list, list],
    *longrangecc: Tuple[list],
    tree: bool = False,
    reverse: bool = False,
    coupletox: bool = False,
) -> Union[List[cp.ndarray], "TreeNetwork"]:
    """
    Generate MPO representing a tight-binding chain of N oscillators.
    
    Args:
        N: Number of oscillators.
        d: Number of Fock states for each oscillator.
        chainparams: A tuple containing [energies], [couplings], [system_coupling].
        longrangecc: Variable number of long-range coupling coefficient lists.
        tree: If True, return a TreeNetwork.
        reverse: If True, reverse the chain.
        coupletox: If True, use non-number conserving coupling.

    Returns:
        A list of MPO tensors or a TreeNetwork.
    """
    
    b = anih(d)
    bd = crea(d)
    n = numb(d)
    u = unitmat(d)
    
    e = chainparams[0]
    t = chainparams[1]

    numlong = len(longrangecc)
    cc = longrangecc
    D1 = 2 + (1 if coupletox else 2) * (1 + numlong)
    D2 = D1 + 1 if coupletox else D1

    H = []

    if coupletox:
        M = cp.zeros((D1, D2, d, d), dtype=cp.complex128)
        
        up_result = up(e[0] * n, t[0] * b, t[0] * bd, *([cp.zeros_like(u)] * numlong), u)
        
        M[D1 - 1, :, :, :] = up_result
        
        dn_result = dn(e[0] * n, *[cc[j][0] * (b + bd) for j in range(numlong)], b + bd, u)
        
        M[:, 0, :, :] = dn_result
        
        for k in range(numlong):
            M[k + 2, k + 3, :, :] = u
        
        H.append(M)
    else:
        M = cp.zeros((D1, D2, d, d), dtype=cp.complex128)
        M[D1 - 1, :, :, :] = up(e[0] * n, t[0] * b, t[0] * bd, u)
        M[:, 0, :, :] = dn(e[0] * n, b, bd, u)
        if numlong > 0:
            raise ValueError("Long-range couplings with non-hermitian coupling not implemented.")
        
        H.append(M)

    for i in range(1, N - 1):
        M = cp.zeros((D2, D2, d, d), dtype=cp.complex128)
        M[D2 - 1, :, :, :] = up(e[i] * n, t[i] * b, t[i] * bd, *([cp.zeros_like(u)] * numlong), u)
        M[:, 0, :, :] = dn(e[i] * n, *[cc[j][i] * (b + bd) for j in range(numlong)], b, bd, u)
        for k in range(numlong):
            M[k + 3, k + 3, :, :] = u
        
        H.append(M)
    
    # Last site
    dn_last = dn(e[N - 1] * n, *[cc[j][N - 1] * (b + bd) for j in range(numlong)], b, bd, u)
    if tree:
        H.append(dn_last)
    else:
        M = cp.zeros((D2, 1, d, d), dtype=cp.complex128)
        M[:, 0, :, :] = dn_last
        
        H.append(M)

    if tree:
        # The last tensor is already in the correct 3D shape for TreeNetwork
        return TreeNetwork.from_sites(H)
    else:
        # Final shape for standard MPO list is (D_left, D_right, d, d)
        if reverse:
            reverse_mpo(H)
        return H

def methyl_blue_mpo_2(e1, e2, delta, N1, N2, N3, d1, d2, d3, S1a1, S2a1, S1a2, S2a2, cparS1S2):
    
    u = unitmat(3)

    c1 = S1a1[2]
    c2 = S2a2[2]
    c3 = cparS1S2[2]

    s1 = unitcol(1, 3)
    s2 = unitcol(0, 3)
    
    s1_op = cp.dot(s1, s1.T)
    s2_op = cp.dot(s2, s2.T)

    # Hs = (e2-e1)*s2*s2' + δ*(s1*s2' + s2*s1') in Julia
    Hs = e2 * s2_op + e1 * s1_op + delta * (cp.dot(s1, s2.T) + cp.dot(s2, s1.T))
    
    M = cp.zeros((1, 4, 4, 3, 3, 3), dtype=cp.complex128)
    
    # M[1,:,1,1,:,:] = up(Hs, c1*s1*s1', s2*s2', u)
    up_result_1 = up(Hs, c1 * s1_op, s2_op, u)
    M[0, :, 0, 0, :, :] = up_result_1
    
    # M[1,1,:,1,:,:] = up(Hs, c2*s2*s2', s1*s1', u)
    up_result_2 = up(Hs, c2 * s2_op, s1_op, u)
    M[0, 0, :, 0, :, :] = up_result_2

    # M[1,1,1,:,:,:] = up(Hs, c3*(s1*s2'+s2*s1'), u)
    cross_term = c3 * (cp.dot(s1, s2.T) + cp.dot(s2, s1.T))
    up_result_3 = up(Hs, cross_term, u)
    M[0, 0, 0, :, :, :] = up_result_3
    
    H = TreeNetwork.from_sites([M])
    
    tree1 = h_bath_chain(N1, d1, S1a1, S2a1, coupletox=True, tree=True)
    tree2 = h_bath_chain(N2, d2, S2a2, S1a2, coupletox=True, tree=True)
    tree3 = h_bath_chain(N3, d3, cparS1S2, coupletox=True, tree=True)
    
    H.add_tree(0, tree1)
    H.add_tree(0, tree2)
    H.add_tree(0, tree3)
    
    return H

def read_chain_coeffs(fdir: str, *params: Union[str, float]) -> List[np.ndarray]:
    """
    Reads chain coefficients from an HDF5 file.

    Args:
        fdir: Path to the HDF5 file.
        params: A sequence of strings or numbers to navigate the HDF5 group structure.

    Returns:
        A list containing the 'e', 't', and 'c' datasets as numpy arrays.
    """
    with h5py.File(fdir, "r") as fid:
        g = fid
        for par in params:
            if isinstance(par, (int, float)):
                found = False
                for key in g.keys():
                    try:
                        if np.isclose(float(key), par):
                            g = g[key]
                            found = True
                            break
                    except ValueError:
                        continue
                if not found:
                    raise KeyError(f"Numeric key '{par}' not found in group '{g.name}'")
            elif isinstance(par, str):
                g = g[par]
            else:
                raise TypeError(f"Unsupported parameter type: {type(par)}")
        
        dat = [g["e"][()], g["t"][()], g["c"][()]]
    return dat 