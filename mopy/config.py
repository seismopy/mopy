"""
A basic configuration.
"""
import mopy.constants as constants


def get_default_param(item, obj=None):
    """
    Return a default parameter if it is defined, else raise Attribute Error.
    """
    # if the object has the attribute return it
    if obj is not None and hasattr(obj, item):
        return getattr(obj, item)
    # else look for it in constants, case insensitive
    try:  # try to get the item from constants, then try upper
        return getattr(constants, item)
    except AttributeError:
        return getattr(constants, item.upper())
