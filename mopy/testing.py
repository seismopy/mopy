"""
Utilities to assist with tests
"""

import numpy as np


def assert_not_nan(obj, none_ok=False):
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
