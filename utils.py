from dataclasses import dataclass
import typing
import math
import pathlib
import pickle
import queue
import threading
import pandas
import numpy as np


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
    columns: typing.Any
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


class AttrDict(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

    def __dir__(self):
        return list(self.keys())


class Channel:
    q: queue.Queue

    def __init__(self):
        self.q = queue.Queue()

    def put(self, v):

        wait_q = self.q.get()
        wait_q.put(v)

    def get(self):
        wait_q = queue.Queue()
        self.q.put(wait_q)

        return wait_q.get()


@dataclass
class Leaf:
    inner: typing.Any

    def __init__(self, x):
        self.inner = x


def fmap(func, o, LeafType=Leaf):
    """This module implements the functor interface: you have a pytree with
    leaves denoted with Leaf. when you apply fmap to such an object, the
    function will be applied to each leaf to reconstruct a tree of the same
    shape.

    This module is useful to write shape-agnostic data transformations.

    """

    if type(o) == dict:
        return {k: fmap(func, v, LeafType=LeafType) for k, v in o.items()}
    elif type(o) == list:
        return [fmap(func, v, LeafType=LeafType) for v in o]
    elif type(o) == LeafType:
        return LeafType(func(o.inner))
    else:
        return o


def collect(o, LeafType=Leaf):
    l = []
    fmap(l.append, o, LeafType)
    return l


def ri_signal(data, period):
    """x ↦ φ(x) such that φ(x) = φ(x + period)"""
    return (
        np.cos(2 * np.pi / period * data),
        np.sin(2 * np.pi / period * data),
    )


def preprocess_time(observations: pandas.DataFrame):
    """Encode the time_of_day, day_of_week and day_of_year scalars onto a circle
    where 0:00 is as similar to 01:00 as it is to 23:00."""
    # TODO: peut-être que day of month ou moon cycle est utile? Je sais pas
    # trop.
    day_r, day_i = ri_signal(np.array(observations.time_time_of_day), 24.0)
    week_r, week_i = ri_signal(np.array(observations.time_day_of_week), 8.0)
    year_r, year_i = ri_signal(np.array(observations.time_day_of_year), 365.25)

    new_cols = pandas.DataFrame(
        {
            "time_of_day_r": day_r,
            "time_of_day_i": day_i,
            "day_of_week_r": week_r,
            "day_of_week_i": week_i,
            "day_of_year_r": year_r,
            "day_of_year_i": year_i,
        }
    )
    observations = observations.drop(
        ["time_time_of_day", "time_day_of_week", "time_day_of_year"], axis=1
    )
    observations = pandas.concat((observations, new_cols), axis=1)
    return observations
