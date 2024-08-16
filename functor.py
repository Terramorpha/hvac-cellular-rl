"""This module implements the functor interface: you have a pytree with leaves
denoted with Leaf. when you apply fmap to such an object, the function will be
applied to each leaf to reconstruct a tree of the same shape.

This module is useful to write shape-agnostic data transformations.

"""

from dataclasses import dataclass
import typing


@dataclass
class Leaf:
    inner: typing.Any

    def __init__(self, x):
        self.inner = x


def fmap(func, o):
    if type(o) == dict:
        return {k: fmap(func, v) for k, v in o.items()}
    elif type(o) == list:
        return [fmap(func, v) for v in o]
    elif type(o) == Leaf:
        return Leaf(func(o.inner))
    else:
        return o


def example1():
    d = {
        "a": Leaf(1),
        "b": 2,
        "c": Leaf(3),
    }
    d2 = fmap(lambda x: x + 1, d)
    print(d)
    print(d2)
