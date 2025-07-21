# cython: language_level=3

cdef class CompareBackend:
    cdef double compare(self, Py_ssize_t i, Py_ssize_t j):
        raise NotImplementedError


cdef class CompareCallBackend(CompareBackend):
    def __init__(self, object o):
        self.callable = o

    cdef double compare(self, Py_ssize_t i, Py_ssize_t j):
        return self.callable(i, j)


cdef class ComparePythonBackend(CompareBackend):
    def __init__(self, object a, object b):
        self.a = a
        self.b = b

    cdef double compare(self, Py_ssize_t i, Py_ssize_t j):
        return self.a[i] == self.b[j]


cdef class CompareStrBackend(CompareBackend):
    def __init__(self, object a, object b):
        self.a = a
        self.b = b

    cdef double compare(self, Py_ssize_t i, Py_ssize_t j):
        return self.a[i] == self.b[j]


cdef class CompareBufferBackend(CompareBackend):
    def __init__(self, const char[:, :] a, const char[:, :] b):
        assert a.shape[1] == b.shape[1]
        self.a = a
        self.b = b

    cdef double compare(self, Py_ssize_t i, Py_ssize_t j):
        cdef:
            Py_ssize_t t
        for t in range(self.a.shape[1]):
            if self.a[i, t] != self.b[j, t]:
                return 0
        return 1


cdef class CompareBufferBackend2D(CompareBackend):
    def __init__(self, const char[:, :, :] a, const char[:, :, :] b, const double[:] weights):
        assert a.shape[2] == b.shape[2]
        assert a.shape[1] == b.shape[1]
        assert a.shape[1] == weights.shape[0]
        self.a = a
        self.b = b
        self.weights = weights

    cdef double compare(self, Py_ssize_t i, Py_ssize_t j):
        cdef:
            Py_ssize_t t, u
            double result = 0
        for u in range(self.weights.shape[0]):
            for t in range(self.a.shape[2]):
                if self.a[i, u, t] != self.b[j, u, t]:
                    break
            else:
                result += self.weights[u]
        return result / self.weights.shape[0]
