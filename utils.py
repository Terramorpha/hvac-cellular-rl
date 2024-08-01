from dataclasses import dataclass
import typing
import math
import pathlib
import pickle


@dataclass
class ProgressIterator:
    name: str
    elements: list

    def __iter__(self):
        n = len(self.elements)
        for i, x in enumerate(self.elements):
            print(f"\r{self.name} {(i+1)/n:1.2f}", end="")
            yield x


@dataclass
class Progress:
    name: str
    elements: typing.Generator

    def __enter__(self):
        return ProgressIterator(self.name, [x for x in self.elements])

    def __exit__(self, exc_type, exc_value, traceback):
        print("")


def until_convergence(f, epsilon):
    x_last = f()
    while True:
        x_new = f()
        if math.fabs(x_new - x_last) < epsilon:
            break
        x_last = x_new


class DataLoader:
    batch_size: int
    columns: any
    length: int

    def __init__(self, *args, batch_size=16):
        self.batch_size = batch_size
        lengths = [len(arg) for arg in args]
        print(lengths)
        if len(lengths) > 0:
            length0 = lengths[0]
        for length in lengths:
            if length != length0:
                raise Exception("args must be same length")

        self.columns = args
        self.length = length0

    def __iter__(self):
        beg = 0
        while beg < self.length:
            end = min(self.length, beg + self.batch_size)
            yield [col[beg:end] for col in self.columns]
            beg = end


def cache(name: str | None = None):
    """Create a decorator to automatically cache the results of a thunk."""

    def func(thunk):
        nonlocal name
        if name is None:
            name = thunk.__name__
        p = pathlib.Path("./pickles") / pathlib.Path(name + ".pkl")
        if not p.is_file():
            val = thunk()
            with open(p, "wb") as f:
                pickle.dump(val, f)
                return val
        else:
            with open(p, "rb") as f:
                val = pickle.load(f)
                return val

    return func
