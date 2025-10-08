import cupy as cp
import numpy as np
import copy
from typing import List, Tuple, Union, Callable

from .treeBasics import (Tree, TreeNetwork, find_head_node, loop_check,
                         path_from_head, find_child, get_leaves, get_bonds,
                         find_bond, set_head_node)
from .tensorOps import qr_general, contract_c, exponentiate_inplace, update_env, apply_h1, apply_h0, lq, set_bond
from .fundamentals import unitvec as unitcol

def orth_centers_mps(A: 'TreeNetwork') -> List[cp.ndarray]:
    """
    Returns the list of the orthogonality centres of A, assumes A is right-normalised.
    """
    B = [cp.copy(s) for s in A.sites] # Deepcopy of sites
    
    from .treeIterators import Traverse
    
    # Dictionary to access sites by id from B.
    B_dict = {i+1: B[i] for i in range(len(B))}

    for id, _ in Traverse(A.tree).items():
        if id == find_head_node(A.tree):
            continue

        par_id = A.tree[id].parent
        dir_to_child = find_bond(A.tree[par_id], id) # Bond index from parent to child

        # QR decomposition of the parent tensor
        AL, C = qr_general(B_dict[par_id], dir_to_child)
        
        # Contract C into the child tensor
        # The parent bond for a child is always the first one (index 0 in Python)
        B_dict[id] = contract_c(B_dict[id], C, 0)

    # Convert dict back to list in original order
    return [B_dict[i+1] for i in range(len(B))]

def to_gpu(net: 'TreeNetwork') -> 'TreeNetwork':
    """
    Convert a TreeNetwork's site tensors to CuPy arrays if they are not already.
    """
    gpu_sites = [None] * len(net.sites)
    for i, site in enumerate(net.sites):
        if isinstance(site, np.ndarray):
            gpu_sites[i] = cp.asarray(site)
        else:
            gpu_sites[i] = site
    return TreeNetwork(net.tree, gpu_sites)


def phys_dims(M: 'TreeNetwork') -> Tuple[int, ...]:
    """
    Return the physical dimensions of a tree-MPS or tree-MPO `M`.
    """
    return tuple(site.shape[-1] for site in M)


def bond_dims(A: 'TreeNetwork') -> np.ndarray:
    """
    Return the bond dimensions of a tree-MPS `A`.
    """
    N = len(A)
    mat = np.zeros((N, N), dtype=int)
    for id1, id2 in get_bonds(A):
        # find_bond returns the index of the dimension corresponding to the bond
        dir_ = find_bond(A.tree[id1], id2)
        # Python uses 0-based indexing for shapes
        D = A[id1].shape[dir_ - 1] # Assumes find_bond is 1-based
        mat[id1 - 1, id2 - 1] = D # Assumes ids are 1-based
        mat[id2 - 1, id1 - 1] = D
    return mat


def mps_right_norm(net: 'TreeNetwork', id: int = -1):
    """
    When applied to a tree-MPS, right normalise towards head-node.
    This is a recursive function.
    """
    if id == -1:
        # Start the process from the head node if no id is given
        loop_check(net.tree)
        id = find_head_node(net)

    node = net.tree[id]
    for i, child_id in enumerate(node.children):
        if net.tree[child_id].children: # If child is not a leaf
            mps_right_norm(net, child_id)
        
        child_tensor = net[child_id]
        
        # Reshape child tensor to a matrix for LQ decomposition
        original_shape = child_tensor.shape
        # The first dimension is the bond to the parent
        matrix_view = child_tensor.reshape(original_shape[0], -1, order='F')
        
        C, AR = lq(matrix_view)

        # Reshape the orthogonal part (AR) back to the original tensor shape
        net[child_id] = AR.reshape(original_shape, order='F')

        # Contract the triangular part (C) into the parent tensor (net[id])
        parent_tensor = net[id]
        
        # This contracts the (i+1)-th index of parent_tensor with the first index of C
        parent_rank = len(parent_tensor.shape)
        
        parent_einsum_indices = list("abcdefghijklmnopqrstuvwxyz"[:parent_rank])
        c_einsum_indices = [parent_einsum_indices[i+1], 'A'] # Contract with (i+1)th index
        result_einsum_indices = parent_einsum_indices[:]
        result_einsum_indices[i+1] = 'A'
        
        einsum_str = f'{"".join(parent_einsum_indices)},{"".join(c_einsum_indices)}->{"".join(result_einsum_indices)}'
        
        new_parent = cp.einsum(einsum_str, parent_tensor, C, optimize=True)
        
        net[id] = new_parent


def mps_mixed_norm(net: 'TreeNetwork', id: int):
    """
    Normalise tree-MPS `A` such that orthogonality centre is on site `id`.
    """
    if not (1 <= id <= len(net)):
        raise IndexError(f"ID {id} is out of bounds for network of length {len(net)}")
    set_head_node(net, id)
    mps_right_norm(net)


def mps_move_oc(A: 'TreeNetwork', id: int):
    """
    Move the orthogonality centre of right normalised tree-MPS `A` to site `id`.
    This function is more efficient than using `mps_mixed_norm` if the tree-MPS is
    already right-normalised.
    """
    path = path_from_head(A.tree, id)
    for site_id in path[1:]: # Drop the head node itself
        mps_shift_oc(A, site_id)


def mps_shift_oc(A: 'TreeNetwork', newhd_id: int):
    """
    Shift the orthogonality centre by one site, setting new head-node `newhd_id`.
    """
    oldhd_id = find_head_node(A)
    if newhd_id not in A.tree[oldhd_id].children:
        raise ValueError(f"Site {newhd_id} is not a child of the head-node {oldhd_id}")

    # The bond to the new head is the first dimension after re-rooting
    set_head_node(A, newhd_id)

    # QR on the old head tensor
    oldhd_tensor = A[oldhd_id]
    original_shape = oldhd_tensor.shape
    # The bond to the new head is now the parent bond, which is the first one
    matrix_view = oldhd_tensor.reshape(original_shape[0], -1, order='F')
    Q, R = cp.linalg.qr(matrix_view, mode='reduced')

    A[oldhd_id] = Q.reshape(original_shape, order='F')

    # Contract R into the new head tensor
    newhd_tensor = A[newhd_id]
    # Find which child of newhd_id is the oldhd_id
    child_idx = find_child(A.tree[newhd_id], oldhd_id)
    
    # Contract C into the (child_idx + 1)-th dimension (parent is dim 0)
    newhd_rank = len(newhd_tensor.shape)
    newhd_indices = list("abcdefghijklmnopqrstuvwxyz"[:newhd_rank])
    r_indices = [newhd_indices[child_idx + 1], 'A']
    result_indices = newhd_indices[:]
    result_indices[child_idx + 1] = 'A'
    einsum_str = f'{"".join(newhd_indices)},{"".join(r_indices)}->{"".join(result_indices)}'
    A[newhd_id] = cp.einsum(einsum_str, newhd_tensor, R)


def _norm_mps_recursive(net: 'TreeNetwork', id: int) -> cp.ndarray:
    """Recursive helper for norm_mps for non-orthogonal states."""
    def get_einsum_str(num_indices: int):
        """Helper to generate an einsum string."""
        return "".join(chr(ord('a') + i) for i in range(num_indices))

    node = net.tree[id]
    site_tensor = net[id]
    rank = len(site_tensor.shape)
    
    # 1. Contract local tensor with its conjugate over the physical index
    indices = get_einsum_str(rank)
    conj_indices = get_einsum_str(rank).upper()
    einsum_str_rho = f"{indices},{conj_indices.replace(conj_indices[-1], indices[-1])}->{indices[:-1]}{conj_indices[:-1]}"
    rho = cp.einsum(einsum_str_rho, site_tensor, cp.conj(site_tensor))
    
    # 2. Recursively contract with rho tensors from children
    rho_indices = list(indices[:-1] + conj_indices[:-1])
    
    for i, child_id in enumerate(node.children):
        rho_child = _norm_mps_recursive(net, child_id)
        
        # Find indices in rho to contract with rho_child
        bond_idx_A = i + 1
        bond_idx_B = i + 1 + (rank - 1)
        
        child_einsum_indices = "uv"
        output_indices = [idx for k, idx in enumerate(rho_indices) if k not in (bond_idx_A, bond_idx_B)]
        
        rho_einsum_list = list(rho_indices)
        rho_einsum_list[bond_idx_A] = child_einsum_indices[0]
        rho_einsum_list[bond_idx_B] = child_einsum_indices[1]
        
        einsum_str_contract = f"{''.join(rho_einsum_list)},{child_einsum_indices}->{''.join(output_indices)}"
        
        rho = cp.einsum(einsum_str_contract, rho, rho_child)
        rho_indices = output_indices

    return rho


def norm_mps(net: 'TreeNetwork', mpsorthog: Union[str, int] = 'None') -> float:
    """
    Calculates the norm of a tree-MPS.
    For non-orthogonal states, it contracts the network from the leaves up.
    """
    loop_check(net.tree)
    if mpsorthog == 'Right' or isinstance(mpsorthog, int):
        OC = mpsorthog if isinstance(mpsorthog, int) else find_head_node(net)
        AC = net[OC]
        norm_sq = cp.vdot(AC, AC)
        return norm_sq.real
    elif mpsorthog == 'None':
        hn = find_head_node(net)
        rho_final = _norm_mps_recursive(net, hn)
        return rho_final.item().real
    else:
        raise ValueError(f"Unsupported value for mpsorthog: {mpsorthog}")


def _calc_bond_dims_recursive(tree: 'Tree', physdims: Tuple[int, ...], Dmax: int, M: np.ndarray, id: int):
    """ Recursive helper for calc_bond_dims """
    for child_id in tree[id].children:
        _calc_bond_dims_recursive(tree, physdims, Dmax, M, child_id)
        D = physdims[child_id] # physdims is 0-indexed
        for grandchild_id in tree[child_id].children:
            D *= M[child_id, grandchild_id]
        
        M[id, child_id] = min(D, Dmax)
        M[child_id, id] = min(D, Dmax)


def calc_bond_dims(tree: 'Tree', physdims: Tuple[int, ...], Dmax: int) -> np.ndarray:
    """
    Calculates required bond dimensions for a given `Dmax`.
    """
    loop_check(tree)
    N = len(tree)
    M = np.zeros((N, N), dtype=int)
    hn = find_head_node(tree)
    _calc_bond_dims_recursive(tree, physdims, Dmax, M, hn)
    return M


def product_state_mps(tree_: 'Tree', physdims_tuple: Tuple[int, ...], Dmax: int = 1, state: Union[str, List[np.ndarray]] = 'Vacuum') -> 'TreeNetwork':
    """
    Return a tree-MPS representing a product state.
    """
    tree = copy.deepcopy(tree_)
    hn = find_head_node(tree)
    leaf_nodes = get_leaves(tree)
    N = len(tree)
    set_head_node(tree, leaf_nodes[0])
    bond_dims1 = calc_bond_dims(tree, physdims_tuple, Dmax)
    set_head_node(tree, leaf_nodes[1])
    bond_dims2 = calc_bond_dims(tree, physdims_tuple, Dmax)
    bond_dims = np.minimum(bond_dims1, bond_dims2)
    tree = copy.deepcopy(tree_)

    if state == 'Vacuum':
        state_list = [unitcol(0, d) for d in physdims_tuple]
    elif state == 'FullOccupy':
        state_list = [unitcol(d - 1, d) for d in physdims_tuple]
    elif isinstance(state, list):
        state_list = state
    else:
        raise ValueError("state input not recognised")

    A = [None] * N
    hn = find_head_node(tree)

    for id, node in enumerate(tree.nodes):
        D_children_dims = [bond_dims[id, child_id] for child_id in node.children]

        if node.parent == -1: # Head-node
            D_par_dim = 1
        else:
            D_par_dim = bond_dims[id, node.parent]
        
        # Shape: (D_parent, D_child1, D_child2, ..., d_phys)
        tensor_shape_dims = [D_par_dim] + D_children_dims
        # In the original Julia implementation a leaf tensor (non-head) only carries the
        # incoming parent bond and its physical index.  It does *not* have an extra dummy
        # bond dimension of size 1.
        
        tensor_shape = (*tensor_shape_dims, physdims_tuple[id])
        
        prod_D_children = int(np.prod(D_children_dims)) if D_children_dims else 1
        B_flat = cp.zeros((D_par_dim, prod_D_children, physdims_tuple[id]), dtype=cp.complex128)

        min_dim = min(D_par_dim, prod_D_children)

        if physdims_tuple[id] > 0:
            state_vec = cp.asarray(state_list[id]).flatten()
            for j in range(min_dim):
                B_flat[j, j, :] = state_vec

        B_tensor = B_flat.reshape(tensor_shape, order='F')

        A[id] = B_tensor

    net = TreeNetwork(tree, A)
    mps_right_norm(net)

    return net

def mps_embed(A: 'TreeNetwork', Dmax: int):
    """
    Embed tree-MPS `A` in manifold of max bond-dimension `Dmax`.
    This function modifies the tensors in `A` in-place.
    """
    tree = copy.deepcopy(A.tree)
    pdims = phys_dims(A)
    hn = find_head_node(tree)
    leaf_nodes = get_leaves(tree)

    set_head_node(tree, leaf_nodes[0])
    bond_dims1 = calc_bond_dims(tree, pdims, Dmax)
    set_head_node(tree, leaf_nodes[1])
    bond_dims2 = calc_bond_dims(tree, pdims, Dmax)
    bond_dims = np.minimum(bond_dims1, bond_dims2)

    for idx, node in enumerate(A.tree.nodes):
        if node.parent != -1:
            parent_dim = bond_dims[idx, node.parent]
        else:
            parent_dim = 1

        child_dims = [bond_dims[idx, child_id] for child_id in node.children]
        target_bond_dims = (parent_dim, *child_dims)
        current_tensor = A[idx]

        A[idx] = set_bond(current_tensor, *target_bond_dims)

    return A


def update_right_env(A: cp.ndarray, M: cp.ndarray, *child_envs: cp.ndarray) -> cp.ndarray:
    return update_env(A, M, *child_envs, dir_open=0, parent_is_left=True)

def update_left_env(A: cp.ndarray, M: cp.ndarray, child_idx: int, parent_env: cp.ndarray, *other_child_envs: cp.ndarray) -> cp.ndarray:
    all_child_envs = list(other_child_envs)
    # The open bond is towards `child_idx`. The parent environment is on the LEFT, therefore we are constructing a LEFT environment.
    fs = [parent_env] + all_child_envs
    return update_env(A, M, *fs, dir_open=child_idx + 1, parent_is_left=False)


def init_envs_recursive(A: 'TreeNetwork', M: 'TreeNetwork', F: List, id: int) -> List:
    for child_id in A.tree[id].children:
        F = init_envs_recursive(A, M, F, child_id)

    # Collect already initialised environments of the children
    child_envs = [F[child_id] for child_id in A.tree[id].children]

    # Store environment for the current node in slot with same index
    F[id] = update_right_env(A[id], M[id], *child_envs)
    return F

def init_envs(A: 'TreeNetwork', M: 'TreeNetwork', F: List = None) -> List:
    """Initialise right-orthonormalized environments."""
    hn_id = find_head_node(A)
    N = len(A)
    F = [None] * N
    
    for child_id in A.tree[hn_id].children:
        F = init_envs_recursive(A, M, F, child_id)
        
    return F


def tdvp1_sweep(dt, A: 'TreeNetwork', M: 'TreeNetwork', F: List = None, timestep_idx: int = 0, **kwargs):
    """
    Main entry point for a 1-site TDVP sweep.
    """
    F = init_envs(A, M, F) # F holds the right environment for each node
    head_id = find_head_node(A)
    return _tdvp1_sweep_recursive(dt, A, M, F, head_id, timestep_idx, **kwargs)


def _tdvp1_sweep_recursive(dt, A: 'TreeNetwork', M: 'TreeNetwork', F: List, id: int, timestep_idx: int = 0, **kwargs):
    children = A.tree[id].children
    parent_id = A.tree[id].parent
    AC = A[id]
    F_parent = cp.ones((1, 1, 1), dtype=cp.complex128) if parent_id == -1 else F[parent_id]
    F_children = [F[child_id] for child_id in children]

    localV = kwargs.get('localV', None)
    def _apply_localV(x):
        if localV is None:
            return x
        Vloc = localV.get(id, None)
        if Vloc is None:
            return x
        # Apply on physical index: reshape (Dl*Dr*... , d) * (d,d)
        x_shape = x.shape
        d = x_shape[-1]
        x_flat = x.reshape((-1, d), order='F')
        x_flat = x_flat @ Vloc
        return x_flat.reshape(x_shape, order='F')

    def h1_op(x):
        Hx = apply_h1(x, M[id], F_parent, *F_children)
        if localV is not None and localV.get(id, None) is not None:
            Vx = _apply_localV(x)
            return Hx + Vx
        return Hx
    
    # --- Forward sweep ---
    exponentiate_inplace(h1_op, -1j * dt / 2, AC)

    for i, child_id in enumerate(children):
        
        AL, C = qr_general(AC, i + 1) # QR decomposition of AC to get C

        # Update environment for sweep down to the child
        other_children_ids = [c for c in children if c != child_id]
        F_other_children = [F[c_id] for c_id in other_children_ids]
        
        # Open bond is towards child i
        F[id] = update_left_env(AL, M[id], i, F_parent, *F_other_children)

        # Evolve C backward
        exponentiate_inplace(lambda x: apply_h0(x, F[id], F[child_id]), 1j * dt / 2, C)
        
        # Absorb C into the child tensor
        A[child_id] = contract_c(A[child_id], C, 0) # Parent bond is always at index 0
        
        A, F = _tdvp1_sweep_recursive(dt, A, M, F, child_id, timestep_idx, **kwargs)
        
        # --- Left sweep ---
        # After returning from recursion, the OC is at the child. Move it back.
        # This involves QR on the child and absorbing C back into our AL.
        
        AR, C_back = qr_general(A[child_id], 0) # QR on parent bond (index 0)
        A[child_id] = AR
        
        # Update the right environment from the child that just returned
        grandchildren_ids = A.tree[child_id].children
        F_grandchildren = [F[gc_id] for gc_id in grandchildren_ids]
        F[child_id] = update_right_env(AR, M[child_id], *F_grandchildren)
        
        # Evolve C_back forward
        exponentiate_inplace(lambda x: apply_h0(x, F[child_id], F[id]), 1j * dt / 2, C_back)
        
        # Absorb C_back into our temporary tensor AL
        AC = contract_c(AL, C_back, i + 1)

    # Final half-step evolution for AC
    def h1_op_final(x):
        # Environments for children have been updated on the backward pass
        F_children_updated = [F[child_id] for child_id in children]
        Hx = apply_h1(x, M[id], F_parent, *F_children_updated)
        if localV is not None and localV.get(id, None) is not None:
            Vx = _apply_localV(x)
            return Hx + Vx
        return Hx
    
    exponentiate_inplace(h1_op_final, -1j * dt / 2, AC)

    A[id] = AC

    return A, F
