import cupy as cp
import numpy as np
from typing import Callable, Tuple

def is_hermitian(op: cp.ndarray, tol: float = 1e-9) -> bool:
    if op.ndim != 2 or op.shape[0] != op.shape[1]:
        return False
    return cp.allclose(op, op.conj().T, atol=tol)

def exponentiate_inplace(
    hamiltonian_op: Callable[[cp.ndarray], cp.ndarray],
    t: complex,
    psi_to_evolve: cp.ndarray,
    tol: float = 1e-12,
    maxiter: int = 250,
) -> Tuple[bool, int, float]:
    """
    Evaluate e ** (t * hamiltonian_op) * psi_to_evolve with a power series, modifying psi_to_evolve in place.
    `hamiltonian_op` can be an arbitrary linear map, implemented as a callable.
    """
    term = psi_to_evolve.copy()
    
    for iter_count in range(maxiter):
        term = hamiltonian_op(term) * t / (iter_count + 1)
        psi_to_evolve += term
        
        # Convergence check
        term_norm = cp.linalg.norm(term)
        psi_norm = cp.linalg.norm(psi_to_evolve)
        
        if psi_norm > tol:
            if term_norm < tol * psi_norm:
                return True, iter_count + 1, term_norm
        elif term_norm < tol:
            return True, iter_count + 1, term_norm
            
    return False, maxiter, cp.linalg.norm(term)

def ac_oac(ac: cp.ndarray, o: cp.ndarray) -> cp.ndarray:
    return cp.einsum('abk,ks,abs->', ac.conj(), o, ac, optimize=True)

def contract_c(a: cp.ndarray, c: cp.ndarray, dir: int) -> cp.ndarray:
    # index strings for einsum
    a_indices = list("abcdefghijklmnopqrstuvwxyz"[:a.ndim])
    c_indices = [a_indices[dir], 'Z']
    
    # Resulting tensor will have new dimension 'Z' at position 'dir'
    result_indices = a_indices[:]
    result_indices[dir] = 'Z'
    
    einsum_str = f'{"".join(a_indices)},{"".join(c_indices)}->{"".join(result_indices)}'
    
    return cp.einsum(einsum_str, a, c.T, optimize=True)

def rho_aa_star(rho, a, indir=None, outdir=None, contract_from=None):
    """
    Contract the RDM with an MPS.
    Parameters
    ----------
    rho : cp.ndarray
        Current reduced density matrix/environment tensor.
    a : cp.ndarray
        MPS tensor. The first index is assumed to be the *parent* (incoming)
        bond, the last index the physical leg; all other indices are children.
    indir, outdir : int | None
        For tree networks one has to state explicitly which branch is being followed.
        * *indir*  – index of the open leg in *a* that is contracted with ρ
        * *outdir* – index of the leg that remains open in the returned ρ.

    contract_from : {'right', 'left', None}
        Convenience keyword. Only the 'right' case is required by measure.py
    """

    rank = a.ndim

    if contract_from is not None:
        if contract_from == 'right':
            if rank == 2:
                # Leaf tensor: A[parent, phys]
                val = cp.einsum('ab,bs,as->', rho, a.conj(), a, optimize=True)
                return val.reshape(1, 1)
            elif rank == 3:
                return rho_aa_star(rho, a, indir=2, outdir=1)
            else:
                raise NotImplementedError("Right-environment contraction is only implemented for rank <= 3 tensors")
        elif contract_from == 'left': # Default
            pass
        else:
            raise ValueError(f"Unknown contract_from value {contract_from!r}")

    if indir is None and outdir is None:
        if rank == 2:
            # A[parent, phys]
            val = cp.einsum('ab,as,bs->', rho, a.conj(), a, optimize=True)
            return val.reshape(1, 1)
        if rank == 3:
            # Chain tensor (parent, child, phys): keep *child* bond
            return rho_aa_star(rho, a, indir=1, outdir=2)
        raise NotImplementedError(
            "For tensors with more than three legs, specify `indir` and `outdir` explicitly.")

    if rank == 2:
        val = cp.einsum('ab,as,bs->', rho, a.conj(), a, optimize=True)
        return val.reshape(1, 1)
    if rank == 3:
        if outdir is None:
             if indir == 1:
                 return cp.einsum('ij,ics,jcs->', rho, a.conj(), a, optimize=True)
             if indir == 2:
                 return cp.einsum('ij,cis,cjs->', rho, a.conj(), a, optimize=True)
        else:
            if indir == 1:
                return cp.einsum('ij,ias,jbs->ab', rho, a.conj(), a, optimize=True)
            if indir == 2:
                return cp.einsum('ij,ais,bjs->ab', rho, a.conj(), a, optimize=True)
    if rank == 4:
        if indir == 1 and outdir == 2:
            return cp.einsum('ij,iacs,jbcs->ab', rho, a.conj(), a, optimize=True)
        if indir == 1 and outdir == 3:
            return cp.einsum('ij,icas,jcbs->ab', rho, a.conj(), a, optimize=True)
        if indir == 2 and outdir == 1:
            return cp.einsum('ij,aics,bjcs->ab', rho, a.conj(), a, optimize=True)
        if indir == 2 and outdir == 3:
            return cp.einsum('ij,cias,cjbs->ab', rho, a.conj(), a, optimize=True)
        if indir == 3 and outdir == 1:
            return cp.einsum('ij,acis,bcjs->ab', rho, a.conj(), a, optimize=True)
        if indir == 3 and outdir == 2:
            return cp.einsum('ij,cais,cbjs->ab', rho, a.conj(), a, optimize=True)
    if rank == 5:
        if indir == 1 and outdir == 2:
            return cp.einsum('ij,iacds,jbkds->ab', rho, a.conj(), a, optimize=True)
        if indir == 1 and outdir == 3:
            return cp.einsum('ij,ikads,jkbds->ab', rho, a.conj(), a, optimize=True)
        if indir == 1 and outdir == 4:
            return cp.einsum('ij,ikdas,jldbs->ab', rho, a.conj(), a, optimize=True)
        if indir == 2 and outdir == 1:
            return cp.einsum('ij,aikds,bjkds->ab', rho, a.conj(), a, optimize=True)
        if indir == 2 and outdir == 3:
            return cp.einsum('ij,kiads,kjbds->ab', rho, a.conj(), a, optimize=True)
        if indir == 2 and outdir == 4:
            return cp.einsum('ij,kidas,kjdbs->ab', rho, a.conj(), a, optimize=True)
        if indir == 3 and outdir == 1:
            return cp.einsum('ij,akids,bkjds->ab', rho, a.conj(), a, optimize=True)
        if indir == 3 and outdir == 2:
            return cp.einsum('ij,kaids,kbjds->ab', rho, a.conj(), a, optimize=True)
        if indir == 3 and outdir == 4:
            return cp.einsum('ij,kdias,kdjbs->ab', rho, a.conj(), a, optimize=True)
        if indir == 4 and outdir == 1:
            return cp.einsum('ij,akdis,bkdjs->ab', rho, a.conj(), a, optimize=True)
        if indir == 4 and outdir == 2:
            return cp.einsum('ij,kadis,kbdjs->ab', rho, a.conj(), a, optimize=True)
        if indir == 4 and outdir == 3:
            return cp.einsum('ij,kdias,kdjbs->ab', rho, a.conj(), a, optimize=True)

    raise NotImplementedError(f"rho_aa_star not implemented for rank {rank} with indir={indir}, outdir={outdir}")

def rho_ab_star(rho, a, b):
    if a.ndim == 2 and b.ndim == 2:
        return cp.einsum('ij,js,is->', rho, a, b.conj(), optimize=True)
    if a.ndim == 3 and b.ndim == 3:
        return cp.einsum('ij,jbs,ias->ab', rho, a, b.conj(), optimize=True)
    raise NotImplementedError(f"rho_ab_star not implemented for ranks {a.ndim}, {b.ndim}")

def rho_aoa_star(rho, a, o, indir, outdir=None):
    rank = a.ndim
    if outdir is None: # Returns scalar
        if rank == 2:
            return cp.einsum('ij,it,ts,js->', rho, a.conj(), o, a, optimize=True)
        if rank == 3:
            if indir == 1:
                return cp.einsum('ij,ict,ts,jcs->', rho, a.conj(), o, a, optimize=True)
            if indir == 2:
                return cp.einsum('ij,cit,ts,cjs->', rho, a.conj(), o, a, optimize=True)
        if rank == 4:
            if indir == 1:
                return cp.einsum('ij,icdt,ts,jcds->', rho, a.conj(), o, a, optimize=True)
            if indir == 2:
                return cp.einsum('ij,cidt,ts,cjds->', rho, a.conj(), o, a, optimize=True)
            if indir == 3:
                return cp.einsum('ij,cdit,ts,cdjs->', rho, a.conj(), o, a, optimize=True)
        if rank == 5:
            if indir == 1:
                return cp.einsum('ij,icdet,ts,jcdes->', rho, a.conj(), o, a, optimize=True)
            if indir == 2:
                return cp.einsum('ij,cidet,ts,cjdes->', rho, a.conj(), o, a, optimize=True)
            if indir == 3:
                return cp.einsum('ij,cdiet,ts,cdjes->', rho, a.conj(), o, a, optimize=True)
            if indir == 4:
                return cp.einsum('ij,cdeit,ts,cdejs->', rho, a.conj(), o, a, optimize=True)

    else: # Returns new rho
        if rank == 3:
            if indir == 1:
                return cp.einsum('ij,iat,ts,jbs->ab', rho, a.conj(), o, a, optimize=True)
            if indir == 2:
                return cp.einsum('ij,ait,ts,bjs->ab', rho, a.conj(), o, a, optimize=True)
        if rank == 4:
            if indir == 1 and outdir == 2:
                return cp.einsum('ij,iact,ts,jbcs->ab', rho, a.conj(), o, a, optimize=True)
            if indir == 1 and outdir == 3:
                return cp.einsum('ij,icat,ts,jcbs->ab', rho, a.conj(), o, a, optimize=True)
            if indir == 2 and outdir == 1:
                return cp.einsum('ij,aict,ts,bjcs->ab', rho, a.conj(), o, a, optimize=True)
            if indir == 2 and outdir == 3:
                return cp.einsum('ij,ciat,ts,cjbs->ab', rho, a.conj(), o, a, optimize=True)
            if indir == 3 and outdir == 1:
                return cp.einsum('ij,acit,ts,bcjs->ab', rho, a.conj(), o, a, optimize=True)
            if indir == 3 and outdir == 2:
                return cp.einsum('ij,cait,ts,cbjs->ab', rho, a.conj(), o, a, optimize=True)
        if rank == 5:
            if indir == 1 and outdir == 2:
                return cp.einsum('ij,iacdt,ts,jbcds->ab', rho, a.conj(), o, a, optimize=True)
            if indir == 1 and outdir == 3:
                return cp.einsum('ij,icadt,ts,jcbds->ab', rho, a.conj(), o, a, optimize=True)
            if indir == 1 and outdir == 4:
                return cp.einsum('ij,icdat,ts,jcdbs->ab', rho, a.conj(), o, a, optimize=True)
            if indir == 2 and outdir == 1:
                return cp.einsum('ij,aicdt,ts,bjcds->ab', rho, a.conj(), o, a, optimize=True)
            if indir == 2 and outdir == 3:
                return cp.einsum('ij,ciadt,ts,cjbds->ab', rho, a.conj(), o, a, optimize=True)
            if indir == 2 and outdir == 4:
                return cp.einsum('ij,cidat,ts,cjdbs->ab', rho, a.conj(), o, a, optimize=True)
            if indir == 3 and outdir == 1:
                return cp.einsum('ij,acidt,ts,bcjds->ab', rho, a.conj(), o, a, optimize=True)
            if indir == 3 and outdir == 2:
                return cp.einsum('ij,caidt,ts,cbjds->ab', rho, a.conj(), o, a, optimize=True)
            if indir == 3 and outdir == 4:
                return cp.einsum('ij,cdiat,ts,cdjbs->ab', rho, a.conj(), o, a, optimize=True)
            if indir == 4 and outdir == 1:
                return cp.einsum('ij,acdit,ts,bcdjs->ab', rho, a.conj(), o, a, optimize=True)
            if indir == 4 and outdir == 2:
                return cp.einsum('ij,cadit,ts,cbdjs->ab', rho, a.conj(), o, a, optimize=True)
            if indir == 4 and outdir == 3:
                return cp.einsum('ij,cdait,ts,cdbjs->ab', rho, a.conj(), o, a, optimize=True)


    raise NotImplementedError(f"rho_aoa_star not implemented for rank {rank} with indir={indir}, outdir={outdir}")


def update_env(a, m, *fs, dir_open, parent_is_left=False):
    num_children = len(fs)
    if num_children == 0: # Leaf node
        return cp.einsum('at,bts,cs->abc', a.conj(), m, a)

    child_indices_a = list(range(a.ndim))
    child_indices_a.pop(dir_open - 1)
    child_indices_a.pop(-1) # physical leg

    if a.ndim == 3 and num_children == 1:
        f = fs[0]
        if parent_is_left: # left sweep
             return cp.einsum('def,adt,bets,cfs->abc', f, a.conj(), m, a, optimize=True)
        else: # right sweep
             return cp.einsum('def,dat,ebts,fcs->abc', f, a.conj(), m, a, optimize=True)

    if a.ndim == 5 and num_children == 3:
        f1, f2, f3 = fs
        if parent_is_left:
            return cp.einsum('def,ghi,jkl,adgjt,behkts,cfils->abc', f1, f2, f3, a.conj(), m, a, optimize=True)
        else:
            # This corresponds to updateleftenv in Julia with permutations.
            # Assumes A is (parent, c1, c2, c3, s), fs = (parent_env, sib1_env, sib2_env)
            f_parent, f_sib1, f_sib2 = fs
            
            if dir_open == 1:
                a_perm = (0, 2, 3, 1, 4)
                m_perm = (0, 2, 3, 1, 4, 5)
            elif dir_open == 2:
                a_perm = (0, 1, 3, 2, 4)
                m_perm = (0, 1, 3, 2, 4, 5)
            elif dir_open == 3:
                a_perm = (0, 1, 2, 3, 4)
                m_perm = (0, 1, 2, 3, 4, 5)
            else:
                raise ValueError(f"Invalid dir_open={dir_open} for update_env with rank 5 tensor")
            
            at = cp.transpose(a, a_perm)
            mt = cp.transpose(m, m_perm) if m.ndim == 6 else m

            return cp.einsum('def,ghi,jkl,dgjat,ehkbts,filcs->abc', f_parent, f_sib1, f_sib2, at.conj(), mt, at, optimize=True)

    raise NotImplementedError


def apply_h1(ac, m, *fs):
    n_fs = len(fs)

    if n_fs == 0:
        raise ValueError("apply_h1 requires at least one environment tensor F.")
    if n_fs == 1:
        return cp.einsum('abc,cs,bps->ap', fs[0], ac, m, optimize=True)
    if n_fs == 2:
        return cp.einsum('abc,def,cfs,beps->adp', fs[0], fs[1], ac, m, optimize=True)
    if n_fs == 3:
        result = cp.einsum('abc,def,ghi,cfis,behps->adgp', fs[0], fs[1], fs[2], ac, m, optimize=True)
        return result
    if n_fs == 4:
        return cp.einsum('abc,def,ghi,jkl,cfils,behkps->adgjp', fs[0], fs[1], fs[2], fs[3], ac, m, optimize=True)

    raise NotImplementedError


def apply_h0(c, fl, fr):
    return cp.einsum('ikj,jl,mkl->im', fl, c, fr)


def set_left_bond(a: cp.ndarray, d_new: int) -> cp.ndarray:
    """Pads or truncates the left bond dimension (axis 0) of a tensor."""
    old_dims = list(a.shape)
    if old_dims[0] == d_new:
        return a
    new_dims = [d_new] + old_dims[1:]
    new_a = cp.zeros(new_dims, dtype=a.dtype)
    
    slicing_new = [slice(None)] * a.ndim
    slicing_old = [slice(None)] * a.ndim
    
    d_copy = min(old_dims[0], d_new)
    slicing_new[0] = slice(0, d_copy)
    slicing_old[0] = slice(0, d_copy)
    
    new_a[tuple(slicing_new)] = a[tuple(slicing_old)].copy()
    return new_a


def set_right_bond(a: cp.ndarray, d_new: int) -> cp.ndarray:
    """Pads or truncates the right bond dimension (axis 1) of a tensor.
    For rank-3 tensors the second axis is the right bond.  For rank-2 leaf
    tensors there is no right bond, so this operation is a no-op.
    """
    old_dims = list(a.shape)

    # If the tensor is rank-2 (Dl × d) we simply return it unchanged because
    # there is no separate right-bond index to resize.
    if a.ndim == 2:
        return a

    if old_dims[1] == d_new:
        return a

    new_dims = [old_dims[0], d_new] + old_dims[2:]
    new_a = cp.zeros(new_dims, dtype=a.dtype)

    slicing_new = [slice(None)] * a.ndim
    slicing_old = [slice(None)] * a.ndim

    d_copy = min(old_dims[1], d_new)
    slicing_new[1] = slice(0, d_copy)
    slicing_old[1] = slice(0, d_copy)

    new_a[tuple(slicing_new)] = a[tuple(slicing_old)].copy()
    return new_a


def set_bond(a, *dims, axis_to_keep_last=True):
    old_dims = a.shape
    phys_dim = [old_dims[-1]] if axis_to_keep_last else []
    num_bonds = a.ndim - 1 if axis_to_keep_last else a.ndim

    if len(dims) != num_bonds:
        raise ValueError(f"Expected {num_bonds} dimensions, but got {len(dims)}")
    
    new_dims = list(dims) + phys_dim
    new_a = cp.zeros(new_dims, dtype=a.dtype)
    slicing = tuple(slice(0, min(old_dim, new_dim)) for old_dim, new_dim in zip(old_dims, new_dims))
    new_a[slicing] = a[slicing]
    return new_a


def evolve_ac(dt, ac, m, fl, fr, energy=False, projerr=False, **kwargs):
    dl_new, _, _ = fl.shape
    dr_new, _, _ = fr.shape
    ac = set_left_bond(set_right_bond(ac, dr_new), dl_new)

    def hamiltonian_op(x):
        return apply_h1(x, m, fl, fr)

    pe_val = cp.linalg.norm(hamiltonian_op(ac)) if projerr else None
    
    # The copy() is important because exponentiate_inplace modifies its argument
    ac_copy = ac.copy().astype(cp.complex128)
    converged, iterations, residual = exponentiate_inplace(hamiltonian_op, -1j * dt, ac_copy, **kwargs)
    ac_new = ac_copy

    e_val = cp.real(cp.vdot(ac_new, apply_h1(ac_new, m, fl, fr))) if energy else None
    
    info = {'converged': converged, 'iterations': iterations, 'residual': residual}
    return ac_new, (e_val, pe_val, info)

def evolve_c(dt, c, fl, fr, energy=False, projerr=False, **kwargs):
    def hamiltonian_op(x):
        return apply_h0(x, fl, fr)
    
    pe_val = cp.linalg.norm(hamiltonian_op(c)) if projerr else None

    # The copy() is important because exponentiate_inplace modifies its argument
    c_copy = c.copy().astype(cp.complex128)
    converged, iterations, residual = exponentiate_inplace(hamiltonian_op, 1j * dt, c_copy, **kwargs)
    c_new = c_copy

    e_val = cp.real(cp.vdot(c_new, apply_h0(c_new, fl, fr))) if energy else None

    info = {'converged': converged, 'iterations': iterations, 'residual': residual}
    return c_new, (e_val, pe_val, info)

def transpose(a: cp.ndarray, dim1: int, dim2: int) -> cp.ndarray:
    """Swaps two dimensions of a tensor."""
    nd = a.ndim
    perm = list(range(nd))
    perm[dim1], perm[dim2] = perm[dim2], perm[dim1]
    return cp.transpose(a, perm)

def qr_general(a: cp.ndarray, i: int) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Performs QR decomposition of a tensor `a` by reshaping it into a matrix.
    The `i`-th index (0-based) becomes the column index of the matrix.
    Note: This should be called with i adjusted from Julia's 1-based indexing (i.e., Julia's i-1)
    """
    dims = a.shape
    nd = a.ndim

    shift = -(i + 1) # We may want to eventually get rid of this 1-based confusion
    perm = tuple((j - shift) % nd for j in range(nd))
    
    permuted_a = cp.transpose(a, perm)
    
    reshaped_input_for_qr = permuted_a.reshape(-1, dims[i], order='F')
    
    q_factor, r_val = cp.linalg.qr(reshaped_input_for_qr, mode='reduced')
    
    new_dims_q = permuted_a.shape[:-1] + (q_factor.shape[1],)
    reshaped_q = q_factor.reshape(new_dims_q, order='F')

    inv_perm = [0] * nd
    for j in range(nd):
        inv_perm[perm[j]] = j
    inv_perm = tuple(inv_perm)
    al_to_return = cp.transpose(reshaped_q, inv_perm)
    
    return al_to_return, r_val

def qr(a: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Performs QR decomposition of a tensor.
    For a 3D tensor (Dl, Dr, d), this decomposes along the Dr dimension.
    """
    if a.ndim == 3: # equivalent to qr_general(a, 1)?
        dl, dr, d = a.shape
        
        # Permute (Dl, Dr, d) → (Dl, d, Dr), then reshape to (Dl*d, Dr)
        permuted_a = cp.transpose(a, (0, 2, 1))
        matrix_for_qr = permuted_a.reshape(dl * d, dr, order='F')
        
        q_factor, r_val = cp.linalg.qr(matrix_for_qr)
        
        # Reshape Q back to (Dl, d, Dr), then permute to (Dl, Dr, d)
        new_dr = r_val.shape[1]
        q_reshaped = q_factor.reshape(dl, d, new_dr, order='F')
        al = cp.transpose(q_reshaped, (0, 2, 1))
        
        return al, r_val
    else: # equivalent to qr_general(a, a.ndim - 1)?
        original_shape = a.shape # (d_1, d_2, ..., d_{N-1}, d_N)
        d_right = original_shape[-1] # d_N
        d_flattened_left = int(np.prod(original_shape[:-1])) # d_1*d_2*...*d_{N-1}
        
        q_factor, r_val = cp.linalg.qr(a.reshape(d_flattened_left, d_right, order='F'))
        
        new_dr = r_val.shape[1] # inner dimension from thin-QR
        al_shape = original_shape[:-1] + (new_dr,) # append new right bond dimension to (d_1, d_2, ..., d_{N-1})
        al = q_factor.reshape(al_shape, order='F') # (d_1*d_2*...*d_{N-1}, d_N_new) → (d_1, d_2, ..., d_{N-1}, d_N_new)
        
        return al, r_val

def lq(a: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]:
    """
    Performs LQ decomposition of a tensor.
    The tensor `a` is reshaped into a matrix by combining the last N-1 dimensions,
    and LQ decomposition is applied.
    """

    original_shape = a.shape
    d_left = original_shape[0]
    d_right_flattened = int(np.prod(original_shape[1:]))

    matrix_for_lq = a.reshape(d_left, d_right_flattened, order='F')
    q_t, r_t = cp.linalg.qr(matrix_for_lq.T, mode='reduced')
    l_val, q_factor = r_t.T, q_t.T

    # Reshape Q back. New left bond dimension will be min(d_left, d_right_flattened)
    new_dl = q_factor.shape[0]
    ar_shape = (new_dl,) + original_shape[1:]
    ar = q_factor.reshape(ar_shape, order='F')

    return l_val, ar

