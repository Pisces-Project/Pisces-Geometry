"""
Generic mixin types.
"""
import numpy as np


class NumpyArithmeticMixin:
    """
    Generic mixin class to support arithmetic operations on numpy arrays.
    """

    def __add__(self, other):
        return np.add(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return np.subtract(self, other)

    def __rsub__(self, other):
        return np.subtract(other, self)

    def __mul__(self, other):
        return np.multiply(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return np.true_divide(self, other)

    def __rtruediv__(self, other):
        return np.true_divide(other, self)

    def __mod__(self, other):
        return np.mod(self, other)

    def __rmod__(self, other):
        return np.mod(other, self)

    def __floordiv__(self, other):
        return np.floor_divide(self, other)

    def __rfloordiv__(self, other):
        return np.floor_divide(other, self)

    def __pow__(self, other):
        return np.power(self, other)

    def __rpow__(self, other):
        return np.power(other, self)

    def __matmul__(self, other):
        return np.matmul(self, other)

    def __rmatmul__(self, other):
        return np.matmul(other, self)

    def __and__(self, other):
        return np.bitwise_and(self, other)

    def __rand__(self, other):
        return np.bitwise_and(other, self)

    def __or__(self, other):
        return np.bitwise_or(self, other)

    def __ror__(self, other):
        return np.bitwise_or(other, self)

    def __xor__(self, other):
        return np.bitwise_xor(self, other)

    def __rxor__(self, other):
        return np.bitwise_xor(other, self)

    def __rshift__(self, other):
        return np.right_shift(self, other)

    def __rrshift__(self, other):
        return np.right_shift(other, self)

    def __lshift__(self, other):
        return np.left_shift(self, other)

    def __rlshift__(self, other):
        return np.left_shift(other, self)

    def __neg__(self):
        return np.negative(self)

    def __abs__(self):
        return np.abs(self)

    def __pos__(self):
        return np.positive(self)

    def __invert(self):
        return np.invert(self)

    def __iadd__(self, other):
        return np.add(self, other, out=self)

    def __isub__(self, other):
        return np.subtract(self, other, out=self)

    def __imul__(self, other):
        return np.multiply(self, other, out=self)

    def __itruediv__(self, other):
        return np.true_divide(self, other, out=self)

    def __imod__(self, other):
        return np.mod(self, other, out=self)

    def __ifloordiv__(self, other):
        return np.floor_divide(self, other, out=self)

    def __ipow__(self, other):
        return np.power(self, other, out=self)

    def __imatmul__(self, other):
        return np.matmul(self, other, out=self)

    def __iand__(self, other):
        return np.bitwise_and(self, other, out=self)

    def __ior__(self, other):
        return np.bitwise_or(self, other, out=self)

    def __ixor__(self, other):
        return np.bitwise_xor(self, other, out=self)

    def __ilshift__(self, other):
        return np.left_shift(self, other, out=self)

    def __irshift__(self, other):
        return np.right_shift(self, other, out=self)
