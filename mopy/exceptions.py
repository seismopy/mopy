"""
Custom exceptions for package
"""
from __future__ import annotations


class DataQualityError(ValueError):
    """ Raised when some data quality checks are failed. """


class NoPhaseInformationError(Exception):
    """ Raised when an event is being added to a ChannelInfo that does not have any phases """
