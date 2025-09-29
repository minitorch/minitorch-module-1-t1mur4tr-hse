"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - inv
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

_max = max

def mul(x: float, y: float):
    return x * y

def id(x: float):
    return x

def inv(x: float):
    return 1. / x

def add(x: float, y: float):
    return x + y

def neg(x: float):
    return -x

def lt(x: float, y: float):
    return x < y

def eq(x: float, y: float):
    return x == y

def max(x: float, y: float):
    return _max(x, y)

def is_close(x: float, y: float):
    return abs(x - y) < 1e-2

def sigmoid(x: float):
    return 1. / (1. + math.exp(-x)) if x >= 0. else math.exp(x) / (1. + math.exp(x))

def relu(x: float):
    return _max(x, 0.)

def log(x: float):
    return math.log(x)

def exp(x: float):
    return math.exp(x)

def log_back(x: float, d: float):
    return d / x

def inv_back(x: float, d: float):
    return -d / x**2

def relu_back(x: float, d: float):
    return d * (x > 0.)


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

_map = map

def map(f: Callable, l: Iterable):
    return list(_map(f, l))

def zipWith(f: Callable, l1: Iterable, l2: Iterable):
    return map(lambda pair: f(*pair), zip(l1, l2))

def reduce(f: Callable, l: Iterable, d: float):
    for v in l:
        d = f(v, d)
    return d

def negList(l: Iterable):
    return map(neg, l)

def addLists(l1: Iterable, l2: Iterable):
    return zipWith(add, l1, l2)

def sum(l: Iterable):
    return reduce(add, l, 0.)

def prod(l: Iterable):
    return reduce(mul, l, 1.)
