"""
Custom exceptions for package
"""


class DataQualityError(ValueError):
    """ Raised when some data quality checks are failed. """


class NoPhaseInformationError(Exception):
    """ Raised when an event is being added to a ChannelInfo that does not have any phases """
