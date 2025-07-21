cdef class ComparisonBackend:
    cdef double compare(self, Py_ssize_t i, Py_ssize_t j)

cdef class ComparisonCallBackend(ComparisonBackend):
    cdef object callable

cdef class ComparisonPythonBackend(ComparisonBackend):
    cdef object a
    cdef object b

cdef class ComparisonStrBackend(ComparisonBackend):
    cdef unicode a
    cdef unicode b

cdef class ComparisonBufferBackend(ComparisonBackend):
    cdef const char[:, :] a
    cdef const char[:, :] b

cdef class ComparisonBufferBackend2D(ComparisonBackend):
    cdef const char[:, :, :] a
    cdef const char[:, :, :] b
    cdef const double[:] weights
