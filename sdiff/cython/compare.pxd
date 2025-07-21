cdef class CompareBackend:
    cdef double compare(self, Py_ssize_t i, Py_ssize_t j)

cdef class CompareCallBackend(CompareBackend):
    cdef object callable

cdef class ComparePythonBackend(CompareBackend):
    cdef object a
    cdef object b

cdef class CompareStrBackend(CompareBackend):
    cdef unicode a
    cdef unicode b

cdef class CompareBufferBackend(CompareBackend):
    cdef const char[:, :] a
    cdef const char[:, :] b

cdef class CompareBufferBackend2D(CompareBackend):
    cdef const char[:, :, :] a
    cdef const char[:, :, :] b
    cdef const double[:] weights
