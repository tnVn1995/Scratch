"""Implementing Id3 Decision Tree Classification Algorithm"""
import pandas as pd
import numpy as np
from typing import List
import math

def entropy(probs: List[float]) -> float:
    """Compute the entropy given a list of class probabilities"""
    return sum(-p * math.log(p, 2) for p in probs if p > 0)

assert entropy([1.0]) == 0, "Wrong implementation!"
assert entropy([0.5, 0.5]) == 1, "Wrong implementation!"


