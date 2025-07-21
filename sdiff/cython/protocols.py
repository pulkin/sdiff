import array

from .compare import (
    ComparisonBackend,
    ComparisonCallBackend,
    ComparisonPythonBackend,
    ComparisonStrBackend,
    ComparisonBufferBackend,
    ComparisonBufferBackend2D,
)


def wrap(data, allow_python: bool = True, allow_k2d: bool = False, k2d_weights=None):
    """
    Figures out the compare protocol from the argument.

    Parameters
    ----------
    data
        The object to wrap into comparison backend. Can be
        - either a pair of array-like objects
        - or a python callable(i, j)
    allow_python
        If set to True, will allow (slow) python kernels for
        comparing python lists, etc.
    allow_k2d
        If set to True, will allow 2D numpy array kernel.
    k2d_weights
        Optional weights for the previous.

    Returns
    -------
    The resulting protocol.
    """
    if isinstance(data, tuple):
        a, b = data
        ta, tb = type(a), type(b)

        if ta == tb:
            if type(a) is str:
                return ComparisonStrBackend(a, b)
            else:
                try:
                    mem_a = memoryview(a)
                    mem_b = memoryview(b)
                except TypeError:
                    pass
                else:
                    if mem_a.ndim != mem_b.ndim:
                        raise ValueError(f"tensors have different dimensionality: {mem_a.ndim} != {mem_b.ndim}")
                    if mem_a.format == mem_b.format and mem_a.itemsize == mem_b.itemsize:
                        if mem_a.ndim == 1:
                            if mem_a.nbytes == 0 or mem_b.nbytes == 0:  # cannot cast
                                return ComparisonBackend()
                            return ComparisonBufferBackend(
                                mem_a.cast('b', shape=[mem_a.shape[0], mem_a.itemsize]),
                                mem_b.cast('b', shape=[mem_b.shape[0], mem_b.itemsize]),
                            )
                        elif mem_a.ndim == 2 and allow_k2d:
                            if mem_a.shape[1] != mem_b.shape[1]:
                                raise ValueError(f"mismatch of the trailing dimension for 2D extension: {mem_a.shape[1]} != {mem_b.shape[1]}")
                            if mem_a.nbytes == 0 or mem_b.nbytes == 0:  # cannot cast
                                return ComparisonBackend()
                            if k2d_weights is None:
                                k2d_weights = array.array('d', [1] * mem_a.shape[1])
                            return ComparisonBufferBackend2D(
                                mem_a.cast('b').cast('b', shape=[*mem_a.shape, mem_a.itemsize]),
                                mem_b.cast('b').cast('b', shape=[*mem_b.shape, mem_b.itemsize]),
                                k2d_weights,
                            )
                        else:
                            raise ValueError(f"unsupported dimensionality of tensors: {mem_a.ndim}")

        if not allow_python:
            raise ValueError("failed to pick a suitable protocol")
        return ComparisonPythonBackend(a, b)
    else:
        return ComparisonCallBackend(data)
