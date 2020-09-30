"""Implementing Id3 Decision Tree Classification Algorithm"""
from typing import List, Any
from collections import Counter
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')
from sklearn import datasets


def entropy(probs: List[float]) -> float:
    """Compute the entropy given a list of class probabilities"""

    return sum(-p * math.log(p, 2) for p in probs if p > 0)


assert entropy([1.0]) == 0, "Wrong implementation!"
assert entropy([0.5, 0.5]) == 1, "Wrong implementation!"

test = [0]


def class_probabilities(labels: List[Any]) -> List[float]:
    """Calculate class probabilities from data labels"""

    total_obs = len(labels)
    probs = [p / total_obs for p in Counter(labels).values()]
    return probs


def info_gain(labels: List[Any]) -> float:
    """Calculate entropy from data"""
    return entropy(class_probabilities(labels))


assert info_gain(['a']) == 0
assert info_gain([True, False]) == 1


def partition_entropy(vars: List[List[Any]]) -> float:
    """Return the partition entropy"""
    total_count = sum(len(var) for var in vars)


def main():
    X, y = datasets.load_boston(return_X_y=True)
    pass

if __name__ == '__main__':
    main()