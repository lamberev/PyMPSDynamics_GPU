import cupy as cp
from typing import List, Tuple

def reverse_mpo(M: List[cp.ndarray]) -> List[cp.ndarray]:
    """Return a reversed copy of MPO M."""
    Mr = [None] * len(M)
    N = len(M)
    for i in range(N):
        Mr[N-i-1] = cp.transpose(M[i], (1, 0, 2, 3))
    return Mr

def phys_dims(M: List[cp.ndarray]) -> Tuple[int, ...]:
    """Return the physical dimensions of an MPS or MPO `M`."""
    return tuple(site.shape[-1] for site in M)

def bond_dims(M: List[cp.ndarray]) -> Tuple[int, ...]:
    """Return bond dimensions of an MPO/MPS."""
    N = len(M)
    res = [0] * (N + 1)
    res[0] = 1
    res[N] = 1
    for i in range(N - 1):
        res[i+1] = M[i].shape[1]
    return tuple(res)
