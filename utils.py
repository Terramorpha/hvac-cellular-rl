from dataclasses import dataclass
import typing
import math


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
