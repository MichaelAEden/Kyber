import functools
import operator
from typing import TypeVar, Generic, Callable, Iterable

T = TypeVar("T")
U = TypeVar("U")


class Matrix(Generic[T]):

    def __init__(self, data: Iterable[Iterable[T]], row_major_order: bool = True):
        col_count = None
        rows = []
        for i, row in enumerate(data):
            row = tuple(row)
            if col_count is not None and len(row) != col_count:
                raise ValueError("All rows and columns must have same length.")
            rows.append(row)
            col_count = len(row)
        row_count = len(rows)
        rows = tuple(rows)

        self._data = rows
        self._row_major_order = row_major_order
        if self._row_major_order:
            self._row_count = row_count
            self._col_count = col_count
        else:
            self._row_count = col_count
            self._col_count = row_count

    @classmethod
    def vector(cls, data: Iterable[T], row_vector: bool = True) -> "Matrix[T]":
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

    def transform(self, func: Callable[[T], U]) -> "Matrix[U]":
        return Matrix(
            (
                (func(self[i, j]) for i in range(self._row_count))
                for j in range(self._col_count)
            ),
            row_major_order=self._row_major_order,
        )

    def __iter__(self):
        for i in range(self._row_count):
            for j in range(self._col_count):
                yield self[i, j]

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

    def _element_wise_operation(
        self, other: "Matrix[T]", func: Callable
    ) -> "Matrix[T]":
        return Matrix(
            (func(self[i, j], other[i, j]) for i in range(self._row_count))
            for j in range(self._col_count)
        )

    def __add__(self, other: "Matrix[T]") -> "Matrix[T]":
        if self.shape != other.shape:
            raise ValueError(
                f"Cannot add matrices of different sizes: {self.shape}, {other.shape}."
            )
        return self._element_wise_operation(other, func=lambda x, y: x + y)

    def __sub__(self, other: "Matrix[T]") -> "Matrix[T]":
        if self.shape != other.shape:
            raise ValueError(
                f"Cannot subtract matrices of different sizes: {self.shape}, {other.shape}."
            )
        return self._element_wise_operation(other, func=lambda x, y: x - y)

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
