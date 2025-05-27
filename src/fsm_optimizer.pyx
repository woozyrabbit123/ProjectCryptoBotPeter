# cython: language_level=3
# cython: cdivision=True
from libc.math cimport fabs

cpdef bint fast_should_wake(double current_price, double current_ema, double threshold_value):
    return fabs(current_price - current_ema) > threshold_value 