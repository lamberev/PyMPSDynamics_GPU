import cupy as cp
import numpy as np
from typing import List, Any
from numbers import Number

def reshape_number(x: Number, *dims: int) -> cp.ndarray:
    """
    Reshape a number `x` into a tensor with `len(dims)` legs that must all be of dimension 1.
    This is a Python translation of `Base.reshape(x::Number, dims...)` from `reshape.jl`.
    """
    if np.prod(dims) != 1:
        raise ValueError(f"new dimensions {dims} must be consistent with array size 1")
    return cp.full(dims, x)

def slice_array(A: cp.ndarray, axis: int) -> List[cp.ndarray]:
    """
    Slices an array A along a given axis, returning a list of subarrays.
    This is a Python translation of `slice(A::AbstractArray, id::Int)` from `reshape.jl`.
    Note that `axis` is 0-indexed, corresponding to Julia's 1-indexed `id`.
    
    Example:
    # A 2x3 matrix
    A = cp.array([[1, 2, 3],
                [4, 5, 6]])

    # Slice along axis 0 (the rows)
    row_slices = slice_array(A, axis=0)
    # Result: [array([1, 2, 3]), array([4, 5, 6])]

    # Slice along axis 1 (the columns)
    col_slices = slice_array(A, axis=1)
    # Result: [array([1, 4]), array([2, 5]), array([3, 6])]
    """
    s = A.shape[axis]
    d = A.ndim
    # Create a list of slices. For each i, the slice is [:, :, ..., i, :, ...]
    # where i is at the specified axis.
    return [A[tuple([slice(None)] * axis + [i] + [slice(None)] * (d - axis - 1))] for i in range(s)]

def rpad_list(A: list, length: int, val: Any = -1.0) -> None:
    """
    Right-pads or truncates a list `A` to a given `length`. This function modifies the list in-place.
    This is a Python translation of `rpad!(A::AbstractArray, len::Int, val)` from `reshape.jl`,
    which operates on a mutable Vector.
    """
    l1 = len(A)
    if length < l1:
        del A[length:]
    elif length > l1:
        A.extend([val] * (length - l1)) 