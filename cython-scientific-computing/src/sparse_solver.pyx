# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from libc.math cimport sin, cos, sqrt

cdef void c_rk4_step(
    double t,
    double* y,
    double dt,
    int n,
    void (*derivs)(double, const double*, double*) nogil
) noexcept nogil:
    cdef double k1[4]
    cdef double k2[4]
    cdef double k3[4]
    cdef double k4[4]
    cdef double y_temp[4]
    cdef int i

    derivs(t, y, k1)

    for i in range(n):
        y_temp[i] = y[i] + 0.5 * dt * k1[i]
    derivs(t + 0.5 * dt, y_temp, k2)

    for i in range(n):
        y_temp[i] = y[i] + 0.5 * dt * k2[i]
    derivs(t + 0.5 * dt, y_temp, k3)

    for i in range(n):
        y_temp[i] = y[i] + dt * k3[i]
    derivs(t + dt, y_temp, k4)

    for i in range(n):
        y[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])

# CSR Sparse Matrix-Vector Multiply (SpMV): y = A * x
def csr_spmv(
    cnp.ndarray[cnp.float64_t, ndim=1] values,
    cnp.ndarray[cnp.int32_t, ndim=1] col_indices,
    cnp.ndarray[cnp.int32_t, ndim=1] row_ptr,
    cnp.ndarray[cnp.float64_t, ndim=1] x,
    int num_rows
):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] y = np.zeros(num_rows, dtype=np.float64)
    cdef int i, j, start_idx, end_idx
    cdef double dot

    for i in range(num_rows):
        start_idx = row_ptr[i]
        end_idx = row_ptr[i + 1]
        dot = 0.0
        for j in range(start_idx, end_idx):
            dot += values[j] * x[col_indices[j]]
        y[i] = dot

    return y
