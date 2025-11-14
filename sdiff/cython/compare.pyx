cdef class ComparisonBackend:
    cdef int compare(self, Py_ssize_t i, Py_ssize_t j):
        raise NotImplementedError

    def __call__(self, i, j):
        return bool(self.compare(i, j))

    def resolve(self, Py_ssize_t i, Py_ssize_t j):
        raise NotImplementedError
