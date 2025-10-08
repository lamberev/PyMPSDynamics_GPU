import cupy as cp
import numpy as np
from .treeBasics import TreeNetwork
from .treeIterators import Path
from .tensorOps import rho_aa_star, rho_aoa_star, is_hermitian

def measure_1site_operator(net: TreeNetwork, op, sites):
    """
    For a Tree, compute the local expectation value of a one-site operator O.
    
    Can be for a single site or a range of sites.
    """
    if isinstance(sites, int):
        return _measure_1site_operator_single(net, op, sites)
    elif isinstance(sites, tuple) and len(sites) == 2:
        return _measure_1site_operator_range(net, op, sites)
    else:
        raise ValueError("`sites` must be an integer or a tuple of two integers.")

def _measure_1site_operator_single(net: TreeNetwork, op, site_id: int):
    """Compute expectation value of a one-site operator `op` for a single site `id`."""
    rho = cp.ones((1, 1), dtype=cp.complex128)
    
    # Contract environment from root to the parent of the target site
    for A, id, direction in Path(net, to_node=site_id)[:-1]:
        rho = rho_aa_star(rho, A, 1, direction)
    
    # Contract with the operator at the target site
    val = rho_aoa_star(rho, net[site_id], op, 1, None)
    
    return val.real if is_hermitian(op) else val

def _measure_1site_operator_range(net: TreeNetwork, op, sites: tuple[int, int]):
    """Compute expectation value of a one-site operator `op` for a range of sites."""
    first_site, last_site = sites
    
    path_between = net.path(first_site, last_site)
    if not path_between:
        raise ValueError("No path found between the specified sites.")
        
    N = len(path_between)
    dtype = cp.float64 if is_hermitian(op) else cp.complex128
    exp_val = cp.empty(N, dtype=dtype)

    # This method requires that the path does not pass through the head node
    if net.find_head_node() in path_between[1:-1]:
        raise NotImplementedError("Measurement across the head node is not implemented.")

    # Determine which end of the path is closer to the root
    path_to_first = net.path(net.find_head_node(), first_site)
    path_to_last = net.path(net.find_head_node(), last_site)

    if len(path_to_first) < len(path_to_last):
        nearest_site = first_site
        farthest_site = last_site
    else:
        nearest_site = last_site
        farthest_site = first_site

    # 1. Contract environment from the root to the nearest site on the path
    rho = cp.ones((1, 1), dtype=cp.complex128)
    for A, id, direction in Path(net, to_node=nearest_site)[:-1]:
        rho = rho_aa_star(rho, A, 1, direction)

    # 2. Iterate along the path, calculating expectation value at each site
    path_iterator = Path(net, from_node=nearest_site, to_node=farthest_site)
    for i, (A, id, direction) in enumerate(path_iterator):
        val = rho_aoa_star(rho, A, op, 1, None)
        exp_val[i] = val.real if dtype == cp.float64 else val
        if direction is not None:
            rho = rho_aa_star(rho, A, 1, direction)
            
    # If we traversed backwards, reverse the results
    if nearest_site == last_site:
        exp_val = exp_val[::-1]
        
    return exp_val

def measure_2site_operator(net: TreeNetwork, op1, op2, sites):
    """
    For a Tree, compute the local expectation value of two one-site operators.
    
    Can be for a range of sites or two specific sites.
    """
    if isinstance(sites, tuple) and len(sites) == 2:
        return _measure_2site_operator_range(net, op1, op2, sites)
    elif isinstance(sites, (list, np.ndarray)) and len(sites) == 2:
         return _measure_2site_operator_specific(net, op1, op2, sites[0], sites[1])
    else:
        raise ValueError("`sites` must be a tuple of two integers for a range, or a list/array of two integers for specific sites.")

def _measure_2site_operator_range(net: TreeNetwork, op1, op2, sites: tuple[int, int]):
    """Computes <O1_i O2_j> for a range of sites i,j."""
    
    # Optimization for O2 = O1'
    if cp.allclose(op2, op1.conj().T):
        return _measure_2site_operator_herm(net, op1, op2, sites)
        
    first_site, last_site = sites
    path_between = net.path(first_site, last_site)
    N = len(path_between)
    
    herm_cis = is_hermitian(op1 @ op2)
    herm_trans = is_hermitian(op1) and is_hermitian(op2)
    dtype = cp.float64 if (herm_cis and herm_trans) else cp.complex128
    exp_val = cp.empty((N, N), dtype=dtype)
    
    if net.find_head_node() in path_between[1:-1]:
        raise NotImplementedError("Measurement across the head node is not implemented.")

    path_to_first = net.path(net.find_head_node(), first_site)
    path_to_last = net.path(net.find_head_node(), last_site)

    if len(path_to_first) < len(path_to_last):
        nearest_site, farthest_site = first_site, last_site
    else:
        nearest_site, farthest_site = last_site, first_site
        
    # Contract environment to the nearest site
    rho = cp.ones((1, 1), dtype=cp.complex128)
    for A, id, dir in Path(net, to_node=nearest_site)[:-1]:
        rho = rho_aa_star(rho, A, 1, dir)
        
    path_iterator = Path(net, from_node=nearest_site, to_node=farthest_site)
    for i, (A_i, id_i, dir_i) in enumerate(path_iterator):
        # Diagonal elements: <(O1*O2)_i>
        exp_val[i, i] = rho_aoa_star(rho, A_i, op1 @ op2, 1, None)
        
        # Propagate environments with O1 and O2 inserted at site i
        rho1 = rho_aoa_star(rho, A_i, op1, 1, dir_i)
        rho2 = rho_aoa_star(rho, A_i, op2, 1, dir_i)

        # Off-diagonal elements: <O1_i O2_j> for j > i
        path_ij = Path(net, from_node=id_i, to_node=farthest_site)
        for j_offset, (A_j, id_j, dir_j) in enumerate(path_ij[1:]):
            j = i + j_offset + 1
            exp_val[i, j] = rho_aoa_star(rho1, A_j, op2, 1, None)
            exp_val[j, i] = rho_aoa_star(rho2, A_j, op1, 1, None)
            if dir_j is not None:
                rho1 = rho_aa_star(rho1, A_j, 1, dir_j)
                rho2 = rho_aa_star(rho2, A_j, 1, dir_j)
        
        if dir_i is not None:
            rho = rho_aa_star(rho, A_i, 1, dir_i)
            
    if dtype == cp.float64:
        exp_val = exp_val.real

    if nearest_site == last_site:
        exp_val = cp.flip(exp_val, axis=(0, 1))
        
    return exp_val

def _measure_2site_operator_herm(net: TreeNetwork, op1, op2, sites: tuple[int, int]):
    """
    Optimized version for two-site measurement where O2 is the hermitian
    conjugate of O1.
    """
    first_site, last_site = sites
    path_between = net.path(first_site, last_site)
    N = len(path_between)
    
    herm_cis = is_hermitian(op1 @ op2)
    herm_trans = is_hermitian(op1) and is_hermitian(op2)
    dtype = cp.float64 if (herm_cis and herm_trans) else cp.complex128
    exp_val = cp.empty((N, N), dtype=dtype)
    
    if net.find_head_node() in path_between[1:-1]:
        raise NotImplementedError("Measurement across the head node is not implemented.")

    path_to_first = net.path(net.find_head_node(), first_site)
    path_to_last = net.path(net.find_head_node(), last_site)
    
    if len(path_to_first) < len(path_to_last):
        nearest_site, farthest_site = first_site, last_site
    else:
        nearest_site, farthest_site = last_site, first_site

    rho = cp.ones((1, 1), dtype=cp.complex128)
    for A, id, dir in Path(net, to_node=nearest_site)[:-1]:
        rho = rho_aa_star(rho, A, 1, dir)

    path_iterator = Path(net, from_node=nearest_site, to_node=farthest_site)
    for i, (A_i, id_i, dir_i) in enumerate(path_iterator):
        exp_val[i, i] = rho_aoa_star(rho, A_i, op1 @ op2, 1, None)
        
        rho1 = rho_aoa_star(rho, A_i, op1, 1, dir_i)
        
        path_ij = Path(net, from_node=id_i, to_node=farthest_site)
        for j_offset, (A_j, id_j, dir_j) in enumerate(path_ij[1:]):
            j = i + j_offset + 1
            val = rho_aoa_star(rho1, A_j, op2, 1, None)
            exp_val[i, j] = val
            exp_val[j, i] = val.conj()
            if dir_j is not None:
                rho1 = rho_aa_star(rho1, A_j, 1, dir_j)

        if dir_i is not None:
            rho = rho_aa_star(rho, A_i, 1, dir_i)

    if dtype == cp.float64:
        exp_val = exp_val.real
        
    if nearest_site == last_site:
        exp_val = cp.flip(exp_val, axis=(0, 1))
        
    return exp_val


def _measure_2site_operator_specific(net: TreeNetwork, op1, op2, site1: int, site2: int):
    """
    Computes <O1_site1 O2_site2>.
    If op1 and op2 have different shapes, computes two separate expectation values.
    """
    if site1 == site2:
        return measure_1site_operator(net, op1 @ op2, site1)

    path_between = net.path(site1, site2)
    if net.find_head_node() in path_between[1:-1]:
        raise NotImplementedError("Measurement across the head node is not implemented.")

    path_to_s1 = net.path(net.find_head_node(), site1)
    path_to_s2 = net.path(net.find_head_node(), site2)

    if len(path_to_s1) < len(path_to_s2):
        nearest_site, farthest_site = site1, site2
        m1, m2 = op1, op2
    else:
        nearest_site, farthest_site = site2, site1
        m1, m2 = op2, op1

    # Contract environment to the nearest site
    rho = cp.ones((1, 1), dtype=cp.complex128)
    for A, id, dir in Path(net, to_node=nearest_site)[:-1]:
        rho = rho_aa_star(rho, A, 1, dir)
    
    # Propagate environment with m1 operator at nearest_site
    path_iterator = Path(net, from_node=nearest_site, to_node=farthest_site)
    A_near, id_near, dir_near = path_iterator[0]
    rho1 = rho_aoa_star(rho, A_near, m1, 1, dir_near)

    # Continue propagating to the operator m2 at farthest_site
    for A, id, dir in path_iterator[1:-1]:
        rho1 = rho_aa_star(rho1, A, 1, dir)
        
    A_far = path_iterator[-1][0]
    val = rho_aoa_star(rho1, A_far, m2, 1, None)

    if is_hermitian(op1) and is_hermitian(op2):
        return val.real
    return val

def measure(net: TreeNetwork, observables: list, light_cone=None):
    """
    Generic measurement function that dispatches to the correct specific function.
    `light_cone` is not used in the tree implementation but included for API consistency.
    """
    if not observables:
        return []
    
    results = [None] * len(observables)
    for k, obs in enumerate(observables):
        if isinstance(obs, dict):
             op = obs.get("op")
             sites = obs.get("sites")
             if "op2" in obs:
                 op2 = obs.get("op2")
                 results[k] = measure_2site_operator(net, op, op2, sites)
             else:
                 results[k] = measure_1site_operator(net, op, sites)
        else:
             # Fallback for simpler cases
             raise TypeError(f"Unsupported observable format: {type(obs)}")

    return results 