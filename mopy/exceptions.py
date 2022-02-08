"""
Custom exceptions for package
"""
from __future__ import annotations


class DataQualityError(ValueError):
    """Raised when some data quality checks are failed."""


class NoPhaseInformationError(Exception):
    """Raised when something does not contain phase information"""
