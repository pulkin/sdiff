cdef class ComparisonBackend:
    cdef double compare(self, Py_ssize_t i, Py_ssize_t j):
        raise NotImplementedError

    def __call__(self, i, j):
        return self.compare(i, j)
