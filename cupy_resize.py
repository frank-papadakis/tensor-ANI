import numpy

import cupy
from cupy import core

_resize_kernel = core.ElementwiseKernel(
    'raw T x, int64 size', 'T y',
    'y = x[i % size]',
    'resize',
)


def resize(a, new_shape):
    """Return a new array with the specified shape.
    If the new array is larger than the original array, then the new
    array is filled with repeated copies of ``a``.  Note that this behavior
    is different from a.resize(new_shape) which fills with zeros instead
    of repeated copies of ``a``.
    Args:
        a (array_like): Array to be resized.
        new_shape (int or tuple of int): Shape of resized array.
    Returns:
        cupy.ndarray:
            The new array is formed from the data in the old array, repeated
            if necessary to fill out the required number of elements.  The
            data are repeated in the order that they are stored in memory.
    .. seealso:: :func:`numpy.resize`
    """
    if numpy.isscalar(a):
        return cupy.full(new_shape, a)
    a = cupy.asarray(a)
    if a.size == 0:
        return cupy.zeros(new_shape, dtype=a.dtype)
    out = cupy.empty(new_shape, a.dtype)
    _resize_kernel(a, a.size, out)
    return out