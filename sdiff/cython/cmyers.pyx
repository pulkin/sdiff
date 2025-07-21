# cython: language_level=3
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from .compare cimport (
    CompareBackend,
    CompareCallBackend,
    ComparePythonBackend,
    CompareStrBackend,
    CompareBufferBackend,
    CompareBufferBackend2D,
)

import array
import cython
from warnings import warn


cdef CompareBackend _get_protocol(Py_ssize_t n, Py_ssize_t m, object compare, int ext_no_python=0, int ext_2d_kernel=0, ext_2d_kernel_weights=None):
    """
    Figures out the compare protocol from the argument.

    Parameters
    ----------
    n, m
        The size of objects being compared.
    compare
        A callable or a tuple of entities to compare.
    ext_no_python
        If set to True, will disallow python protocols
        (``__eq__`` and call) but raise instead.
    ext_2d_kernel
        If set to True, will allow 2D numpy array kernel.
    ext_2d_kernel_weights
        Optional weights for the previous.

    Returns
    -------
    The resulting protocol.
    """
    if isinstance(compare, tuple):
        a, b = compare
        ta, tb = type(a), type(b)

        if ta == tb:
            if type(a) is str:
                return CompareStrBackend(a, b)
            else:
                try:
                    mem_a = memoryview(a)
                    mem_b = memoryview(b)
                except:
                    pass
                else:
                    if mem_a.ndim != mem_b.ndim:
                        raise ValueError(f"tensors have different dimensionality: {mem_a.ndim} != {mem_b.ndim}")
                    if mem_a.format == mem_b.format and mem_a.itemsize == mem_b.itemsize:
                        if mem_a.ndim == 1:
                            if mem_a.nbytes == 0 or mem_b.nbytes == 0:  # cannot cast
                                return CompareBackend()
                            return CompareBufferBackend(
                                mem_a.cast('b', shape=[mem_a.shape[0], mem_a.itemsize]),
                                mem_b.cast('b', shape=[mem_b.shape[0], mem_b.itemsize]),
                            )
                        elif mem_a.ndim == 2 and ext_2d_kernel:
                            if mem_a.shape[1] != mem_b.shape[1]:
                                raise ValueError(f"mismatch of the trailing dimension for 2D extension: {mem_a.shape[1]} != {mem_b.shape[1]}")
                            if mem_a.nbytes == 0 or mem_b.nbytes == 0:  # cannot cast
                                return CompareBackend()
                            if ext_2d_kernel_weights is None:
                                ext_2d_kernel_weights = array.array('d', [1] * mem_a.shape[1])
                            return CompareBufferBackend2D(
                                mem_a.cast('b').cast('b', shape=[*mem_a.shape, mem_a.itemsize]),
                                mem_b.cast('b').cast('b', shape=[*mem_b.shape, mem_b.itemsize]),
                                ext_2d_kernel_weights,
                            )
                        else:
                            raise ValueError(f"unsupported dimensionality of tensors: {mem_a.ndim}")

        if ext_no_python:
            raise ValueError("failed to pick a suitable protocol")
        return ComparePythonBackend(a, b)
    else:
        return CompareCallBackend(compare)


cdef inline Py_ssize_t labs(long i) noexcept:
    return i if i >= 0 else -i


@cython.cdivision
cdef inline Py_ssize_t _get_diag_index(Py_ssize_t diag, Py_ssize_t nm) noexcept:
    """Computes the index of a given diagonal"""
    return (diag // 2) % nm


# coordinate transformation
# progress = x + y
# diag = x - y + m
#                  0    0                     progress
#         ----------- ◉ ---------------     < 0
#         |    1    ↙   ↘   1         |
#         |       ◉       ◉           |     < 1
#         |2    ↙   ↘   ↙   ↘   2     |
#         |   ◉       ◉       ◉       |     < 2
#      3  | ↙   ↘   ↙   ↘   ↙   ↘   3 |
#         ◉       ◉       ◉       ◉   |     < 3
#         | ↘   ↙   ↘   ↙   ↘   ↙   ↘ | 4
#         |   ◉       ◉       ◉       ◉     < 4
#         |     ↘   ↙   ↘   ↙   ↘   ↙ |
#         |       ◉       ◉       ◉   |     < 5
#         |         ↘   ↙   ↘   ↙     |
#         |           ◉       ◉       |     < 6
#         |             ↘   ↙         |
#         --------------- ◉ -----------     < 7
#
#         ^   ^   ^   ^   ^   ^   ^   ^
#         0   1   2   3   4   5   6   7       diagonal


@cython.cdivision
cdef inline Py_ssize_t _get_x(Py_ssize_t diag, Py_ssize_t progress, Py_ssize_t m) noexcept:
    return (progress + diag - m) // 2


@cython.cdivision
cdef inline Py_ssize_t _get_y(Py_ssize_t diag, Py_ssize_t progress, Py_ssize_t m) noexcept:
    return (progress - diag + m) // 2


cdef inline void _fill_no_solution(char[::1] out, Py_ssize_t i, Py_ssize_t  j, Py_ssize_t n, Py_ssize_t m) noexcept:
    cdef Py_ssize_t ix
    for ix in range(i + j, i + j + n):
        out[ix] = 1
    for ix in range(i + j + n, i + j + n + m):
        out[ix] = 2


cdef inline int _is_left_edge(
        Py_ssize_t diag_updated_from,
        Py_ssize_t n,
        Py_ssize_t m,
        Py_ssize_t nm,
        Py_ssize_t is_reverse_front,
        Py_ssize_t* fronts,
        Py_ssize_t front_updated_offset,
) noexcept:
    cdef Py_ssize_t progress = fronts[front_updated_offset + _get_diag_index(diag_updated_from, nm)]
    if is_reverse_front:
        return _get_x(diag_updated_from, progress, m) == 0
    else:
        return _get_y(diag_updated_from, progress, m) == m


cdef inline int _is_right_edge(
        Py_ssize_t diag_updated_to,
        Py_ssize_t n,
        Py_ssize_t m,
        Py_ssize_t nm,
        Py_ssize_t is_reverse_front,
        Py_ssize_t* fronts,
        Py_ssize_t front_updated_offset,
) noexcept:
    cdef Py_ssize_t progress = fronts[front_updated_offset + _get_diag_index(diag_updated_to, nm)]
    if is_reverse_front:
        return _get_y(diag_updated_to, progress, m) == 0
    else:
        return _get_x(diag_updated_to, progress, m) == n


cdef inline void _do_branch(
        Py_ssize_t nm,
        Py_ssize_t diag_updated_from,
        Py_ssize_t no_branch_left,
        Py_ssize_t diag_updated_to,
        Py_ssize_t no_branch_right,
        Py_ssize_t is_reverse_front,
        Py_ssize_t reverse_as_sign,
        Py_ssize_t* fronts,
        Py_ssize_t front_updated_offset,
) noexcept:
    cdef:
        int ix = -1
        int previous = -1
        int ix0 = _get_diag_index(diag_updated_from, nm)
        int progress0 = fronts[front_updated_offset + ix0]
        Py_ssize_t diag, progress_left, progress_right

    for diag in range(diag_updated_from, diag_updated_to + 2, 2):

        # source and destination indexes for the update
        progress_left = fronts[front_updated_offset + _get_diag_index(diag - 1, nm)]
        if diag == diag_updated_to and _get_diag_index(diag + 1, nm) == ix0:
            progress_right = progress0
        else:
            progress_right = fronts[front_updated_offset + _get_diag_index(diag + 1, nm)]

        if diag == diag_updated_from and not no_branch_left:  # possible in cases 2, 4
            progress = progress_right
        elif diag == diag_updated_to and not no_branch_right:  # possible in cases 1, 3
            progress = progress_left
        elif is_reverse_front:
            progress = min(progress_left, progress_right)
        else:
            progress = max(progress_left, progress_right)

        # the idea here is to delay updating the front by one iteration
        # such that the new progress values do not interfer with the original ones
        if ix != -1:
            fronts[front_updated_offset + ix] = previous + reverse_as_sign

        previous = progress
        ix = _get_diag_index(diag, nm)

    fronts[front_updated_offset + ix] = previous + reverse_as_sign


@cython.cdivision
cdef Py_ssize_t _search_graph_recursive(
    Py_ssize_t n,
    Py_ssize_t m,
    CompareBackend compare_backend,
    const double accept,
    Py_ssize_t max_cost,
    Py_ssize_t max_calls,
    char eq_only,
    char[::1] out,
    Py_ssize_t i,
    Py_ssize_t j,
    Py_ssize_t* fronts,
    Py_ssize_t* front_ranges,
):
    """See the description and details in the pure-python implementation"""
    cdef:
        Py_ssize_t ix, nm, n_m, cost, diag, diag_src, diag_dst, diag_facing_from, diag_facing_to, diag_updated_from,\
            diag_updated_to, diag_, diag_updated_from_, diag_updated_to_, x, y, x2, y2, progress, progress_start,\
            previous, is_reverse_front, reverse_as_sign, n_calls = 2, front_updated_offset
        int rtn_script = out.shape[0] != 0, is_left_edge, is_right_edge

    max_cost = min(max_cost, n + m)

    # strip matching ends of the sequence
    # forward
    while n * m > 0 and compare_backend.compare(i, j) >= accept:
        n_calls += 1
        ix = i + j
        if rtn_script:
            out[ix] = 3
            out[ix + 1] = 0
        i += 1
        j += 1
        n -= 1
        m -= 1
    # ... and reverse
    while n * m > 0 and compare_backend.compare(i + n - 1, j + m - 1) >= accept:
        n_calls += 1
        ix = i + j + n + m - 2
        if rtn_script:
            out[ix] = 3
            out[ix + 1] = 0
        n -= 1
        m -= 1

    if n * m == 0:
        if rtn_script:
            _fill_no_solution(out, i, j, n, m)
        return n + m

    nm = min(n, m) + 1
    n_m = n + m
    for ix in range(nm):
        fronts[ix] = 0
    for ix in range(nm, 2 * nm):
        fronts[ix] = n_m

    front_ranges[0] = front_ranges[1] = m
    front_ranges[2] = front_ranges[3] = n

    # we, effectively, iterate over the cost itself
    # though it may also be seen as a round counter
    for cost in range(max_cost + 1):

        # first, figure out whether step is reverse or not
        is_reverse_front = cost % 2
        reverse_as_sign = 1 - 2 * is_reverse_front  # +- 1 depending on the direction

        # one of the fronts is updated, another one we "face"
        front_updated_offset = nm * is_reverse_front
        diag_updated_from = front_ranges[2 * is_reverse_front]
        diag_updated_to = front_ranges[2 * is_reverse_front + 1]
        diag_facing_from = front_ranges[2 * (1 - is_reverse_front)]
        diag_facing_to = front_ranges[2 * (1 - is_reverse_front) + 1]

        # phase 1: propagate diagonals
        # every second diagonal is propagated during each iteration
        for diag in range(diag_updated_from, diag_updated_to + 2, 2):
            # we simply use modulo size for indexing
            # you can also keep diag_from to always correspond to the 0th
            # element of the front or any other alignment but having
            # modulo is just the simplest
            ix = _get_diag_index(diag, nm)

            # remember the progress coordinates: starting, current
            progress = progress_start = fronts[front_updated_offset + ix]

            # now, turn (diag, progress) coordinates into (x, y)
            # progress = x + y
            # diag = x - y + m
            # since the (x, y) -> (x + 1, y + 1) diag is polled through similarity_ratio_getter(x, y)
            # we need to shift the (x, y) coordinates when reverse
            x = _get_x(diag, progress, m) - is_reverse_front
            y = _get_y(diag, progress, m) - is_reverse_front

            # slide down the progress coordinate
            while 0 <= x < n and 0 <= y < m:
                n_calls += 1
                if compare_backend.compare(x + i, y + j) < accept:
                    break
                progress += 2 * reverse_as_sign
                x += reverse_as_sign
                y += reverse_as_sign
            else:
                # we need to adjust one of the edges
                if is_reverse_front:
                    if x == -1:
                        diag_updated_from = max(diag_updated_from, diag)
                    if y == -1:
                        diag_updated_to = min(diag_updated_to, diag)
                else:
                    if x == n:
                        diag_updated_to = min(diag_updated_to, diag)
                    if y == m:
                        diag_updated_from = max(diag_updated_from, diag)
            fronts[front_updated_offset + ix] = progress

            # if front and reverse overlap we are done
            # to figure this out we first check whether we are facing ANY diagonal
            if diag_facing_from <= diag <= diag_facing_to and (diag - diag_facing_from) % 2 == 0:
                # second, we are checking the progress
                if fronts[ix] >= fronts[ix + nm]:  # check if the two fronts (start) overlap
                    if rtn_script:
                        # write the diagonal
                        # cython does not support range(a, b, c)
                        # (probably because of the unknown sign of c)
                        # so use "while"
                        ix = progress_start - 2 * is_reverse_front
                        while ix != progress - 2 * is_reverse_front:
                            out[i + j + ix] = 3
                            out[i + j + ix + 1] = 0
                            ix += 2 * reverse_as_sign

                        # recursive calls
                        x = _get_x(diag, progress_start, m)
                        y = _get_y(diag, progress_start, m)
                        x2 = _get_x(diag, progress, m)
                        y2 = _get_y(diag, progress, m)
                        if is_reverse_front:
                            # swap these two around
                            x, y, x2, y2 = x2, y2, x, y

                        _search_graph_recursive(
                            n=x,
                            m=y,
                            compare_backend=compare_backend,
                            accept=accept,
                            max_cost=cost // 2 + cost % 2,
                            max_calls=max_calls,
                            eq_only=0,
                            out=out,
                            i=i,
                            j=j,
                            fronts=fronts,
                            front_ranges=front_ranges,
                        )
                        _search_graph_recursive(
                            n=n - x2,
                            m=m - y2,
                            compare_backend=compare_backend,
                            accept=accept,
                            max_cost=cost // 2,
                            max_calls=max_calls,
                            eq_only=0,
                            out=out,
                            i=i + x2,
                            j=j + y2,
                            fronts=fronts,
                            front_ranges=front_ranges,
                        )
                    return cost

        if n_calls > max_calls:
            break

        # check left edge
        is_left_edge = _is_left_edge(diag_updated_from, n, m, nm, is_reverse_front, fronts, front_updated_offset)
        diag_updated_from += 2 * is_left_edge - 1
        front_ranges[2 * is_reverse_front] = diag_updated_from

        # check right edge
        is_right_edge = _is_right_edge(diag_updated_to, n, m, nm, is_reverse_front, fronts, front_updated_offset)
        diag_updated_to -= 2 * is_right_edge - 1
        front_ranges[2 * is_reverse_front + 1] = diag_updated_to

        _do_branch(nm, diag_updated_from, is_left_edge, diag_updated_to, is_right_edge, is_reverse_front,
                   reverse_as_sign, fronts, front_updated_offset)

    if rtn_script:
        _fill_no_solution(out, i, j, n, m)
    return n + m


_null_script = array.array('b', b'')


def search_graph_recursive(
    Py_ssize_t n,
    Py_ssize_t m,
    similarity_ratio_getter,
    out=None,
    double accept=1,
    Py_ssize_t max_cost=0xFFFFFFFF,
    Py_ssize_t max_calls=0xFFFFFFFF,
    char eq_only=0,
    Py_ssize_t i=0,
    Py_ssize_t j=0,
    int ext_no_python=0,
    int ext_2d_kernel=0,
    ext_2d_kernel_weights=None,
) -> int:
    """See the description of the pure-python implementation."""
    cdef:
        char[::1] cout
        Py_ssize_t nm = min(n, m) + 1
        Py_ssize_t* buffer = <Py_ssize_t *>PyMem_Malloc(2 * sizeof(Py_ssize_t) * nm)
        Py_ssize_t* buffer2 = <Py_ssize_t *>PyMem_Malloc(2 * sizeof(Py_ssize_t) * 2)
        CompareBackend compare_backend

    if out is None:
        cout = _null_script
    else:
        cout = out
        if eq_only:
            warn("the 'out' argument is ignored for eq_only=True")

    try:
        compare_backend = _get_protocol(n, m, similarity_ratio_getter, ext_no_python, ext_2d_kernel, ext_2d_kernel_weights)
        return _search_graph_recursive(
            n=n,
            m=m,
            compare_backend=compare_backend,
            accept=accept,
            max_cost=max_cost,
            max_calls=max_calls,
            eq_only=eq_only,
            out=cout,
            i=i,
            j=j,
            fronts=buffer,
            front_ranges=buffer2,
        )
    finally:
        PyMem_Free(buffer)
        PyMem_Free(buffer2)
