"""
Utilities to assist with tests
"""
from __future__ import annotations
from typing import Any

import numpy as np


def assert_not_nan(obj: Any, none_ok: bool = False):
    """
    Verify that the passed object is not NaN

    Parameters
    ----------
    obj
        The object to check (can be any python object)
    none_ok
        Flag to indicate whether passing a None object should raise. If False, a None will raise an AssertionError.
        Default is False.

    Raises
    ------
    AssertionError if the obj is NaN-like, or if none_ok is False and the object is None.
    """
    if not none_ok and not obj:
        raise AssertionError()
    try:
        assert not np.isnan(obj)
    except TypeError:
        pass


def gauss(t: np.array, a: float, b: float, c: float):
    """
    Returns a Gaussian wave pulse (ex. for calculating spectra)

    Parameters
    ----------
    t
        Time range over which to generate the wave pulse
    a
        Amplitude of the pulse
    b
        Time at which the peak of the pulse is observed
    c
        Term describing the width of the pulse (larger values result in a wider
        pulse)
    """
    denom = 2 * c ** 2
    exp = -1 * (t - b) ** 2 / denom
    return a * np.exp(exp)


def gauss_deriv(t: np.array, a: float, b: float, c: float):
    """
    Returns the derivative of a Gaussian wave pulse (ex. for calculating spectra)

    Parameters
    ----------
    t
        Time range over which to generate the wave pulse
    a
        Amplitude of the pulse
    b
        Time at which the peak of the pulse is observed
    c
        Term describing the width of the pulse (larger values result in a wider
        pulse)
    """
    denom = 2 * c ** 2
    exp = -1 * (t - b) ** 2 / denom
    fact = 2 * t - 2 * b
    return -a * fact * np.exp(exp) / denom


def gauss_deriv_deriv(t: np.array, a: float, b: float, c: float):
    """
    Returns the second derivative of a Gaussian wave pulse (ex. for calculating spectra)

    Parameters
    ----------
    t
        Time range over which to generate the wave pulse
    a
        Amplitude of the pulse
    b
        Time at which the peak of the pulse is observed
    c
        Term describing the width of the pulse (larger values result in a wider
        pulse)
    """
    denom = 2 * c ** 2
    exp = -1 * (t - b) ** 2 / denom
    fact = 2 * t - 2 * b
    return -a * np.exp(exp) / (denom / 2) + a * fact ** 2 * np.exp(exp) / denom ** 2
