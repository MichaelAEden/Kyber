import functools
import operator
from typing import TypeVar, Generic, Callable

T = TypeVar("T")


class Matrix(Generic[T]):

    # TODO: use iterable rather than tuple type.
    def __init__(self, data: tuple[tuple[T, ...], ...], row_major_order: bool = True):
        k = len(data[0])
        l = len(data)
        for row in data[1:]:
            if len(row) != k:
                raise ValueError("All rows and columns must have same length.")

        self._data = data
        self._row_major_order = row_major_order
        if self._row_major_order:
            self._col_count = k
            self._row_count = l
        else:
            self._col_count = l
            self._row_count = k

    @classmethod
    def vector(cls, data: tuple[T, ...], row_vector: bool = True) -> "Matrix[T]":
        return cls((data,), row_major_order=row_vector)

    @classmethod
    def scalar(cls, data: T) -> "Matrix[T]":
        return cls(((data,),))

    @property
    def shape(self) -> tuple[int, int]:
        return self._row_count, self._col_count

    @property
    def transpose(self) -> "Matrix[T]":
        return Matrix(self._data, row_major_order=not self._row_major_order)

    # TODO: prevent mutation of matrix elements.
    def __hash__(self) -> int:
        # TODO: set PYTHONHASHSEED for consistent hashing behaviour.
        return hash((self._data, self._row_major_order))

    def __eq__(self, other: "Matrix[T]") -> bool:
        return (
            self.shape == other.shape
            and sum(
                self[i, j] != other[i, j]
                for i in range(self._row_count)
                for j in range(self._col_count)
            )
            == 0
        )

    def __getitem__(self, subscript: tuple[int, int]) -> T:
        row, col = subscript
        if row > self._row_count or col > self._col_count:
            raise IndexError(
                f"Index ({row}, {col}) out of range for matrix of size: {self.shape}"
            )

        if self._row_major_order:
            return self._data[row][col]
        else:
            return self._data[col][row]

    def _element_wise_operation(self, other: "Matrix[T]", fn: Callable) -> "Matrix[T]":
        output_rows = []
        for row_i in range(self._row_count):
            output_row = []
            for col_i in range(other._col_count):
                output_row.append(fn(self[row_i, col_i], other[row_i, col_i]))
            output_rows.append(tuple(output_row))
        return Matrix(tuple(output_rows))

    def __add__(self, other: "Matrix[T]") -> "Matrix[T]":
        if self.shape != other.shape:
            raise ValueError(
                f"Cannot add matrices of different sizes: {self.shape}, {other.shape}."
            )
        return self._element_wise_operation(other, fn=lambda x, y: x + y)

    def __sub__(self, other: "Matrix[T]") -> "Matrix[T]":
        if self.shape != other.shape:
            raise ValueError(
                f"Cannot subtract matrices of different sizes: {self.shape}, {other.shape}."
            )
        return self._element_wise_operation(other, fn=lambda x, y: x - y)

    def __mul__(self, other: "Matrix[T]") -> "Matrix[T]":
        if self._col_count != other._row_count:
            raise ValueError(
                f"Cannot multiply matrices with incompatible sizes: {self.shape}, {other.shape}."
            )

        output_rows = []
        for row_i in range(self._row_count):
            output_row = []
            for col_i in range(other._col_count):
                dot_product = functools.reduce(
                    operator.add,
                    (self[row_i, i] * other[i, col_i] for i in range(self._col_count)),
                )
                output_row.append(dot_product)
            output_rows.append(tuple(output_row))

        return Matrix(tuple(output_rows))
