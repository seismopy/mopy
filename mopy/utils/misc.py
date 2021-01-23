"""
Utility functions
"""
import functools
import importlib
import inspect
from numbers import Number
import warnings
from types import ModuleType
from typing import Union, Mapping, Callable, Hashable, Optional, Iterable, List

import numpy as np
import pandas as pd
from obsplus.constants import NSLC
from typing_extensions import Literal

from mopy.constants import _INDEX_NAMES, BroadcastableFloatType


def _func_and_kwargs_str(func: Callable, *args, **kwargs) -> str:
    """
    Get a str rep of the function and input args.
    """
    callargs = inspect.getcallargs(func, *args, **kwargs)
    callargs.pop("self")
    kwargs_ = callargs.pop("kwargs", {})
    arguments = []
    arguments += [f"{k}={repr(v)}" for k, v in callargs.items() if v is not None]
    arguments += [f"{k}={repr(v)}" for k, v in kwargs_.items() if v is not None]
    arguments.sort()
    out = f"{func.__name__}::"
    if arguments:
        out += f"{':'.join(arguments)}"
    return out


def _track_method(idempotent: Union[Callable, bool] = False):
    """
    Keep track of the method call and params.
    """

    def _deco(func, idempotent=idempotent):
        @functools.wraps(func)
        def _wrap(self, *args, **kwargs):
            # if the method is idempotent and already has been called return self
            if idempotent and any(x.startswith(func.__name__) for x in self.processing):
                return self.copy()

            info_str = _func_and_kwargs_str(func, self, *args, **kwargs)
            new = func(self, *args, **kwargs)
            new.processing = tuple(list(new.processing) + [info_str])
            return new

        return _wrap

    # this allows the decorator to be used with or without calling it.
    if callable(idempotent):
        return _deco(idempotent, idempotent=False)
    else:
        return _deco


# --- Misc.


def optional_import(module_name: str) -> ModuleType:
    """
    Try to import a module by name and return it. If unable, raise import error.

    Parameters
    ----------
    module_name
        The name of the module.

    Returns
    -------
    The module object.
    """
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        caller_name = inspect.stack()[1].function
        msg = (
            f"{caller_name} requires the module {module_name} but it "
            f"is not installed."
        )
        raise ImportError(msg)
    return mod


# --- Miscellaneous functions and decorators


def expand_seed_id(seed_id: Union[pd.Series, pd.Index]) -> pd.DataFrame:
    """
    Take a Series of seed_ids and expand to a DataFrame of NSLC

    Parameters
    ----------
    seed_id
        Series of seed_ids

    Returns
    -------
    nslc
        DataFrame of the expanded NSLC
    """
    seed_id_map = {num: code for num, code in enumerate(NSLC)}
    seed_id = pd.Series(seed_id)
    res = seed_id.str.split(".", expand=True).rename(columns=seed_id_map)
    if not len(res.columns) == 4:
        raise ValueError("Provided values are not valid seed ids")
    return res


def pad_or_trim(array: np.ndarray, sample_count: int, pad_value: int = 0) -> np.ndarray:
    """
    Pad or trim an array to a specified length along the last dimension.

    Parameters
    ----------
    array
        A non-empty numpy array.
    sample_count
        The sample count to trim or pad to. If greater than the length of the
        arrays' last dimension the array will be padded with pad_value, else
        it will be trimmed.
    pad_value
        If padding is to occur, the value used to pad the array.
    Returns
    -------
    The trimmed or padded array.
    """
    last_dim_len = np.shape(array)[-1]
    # the trim case
    if sample_count <= last_dim_len:
        return array[..., :sample_count]
    # the fill case
    npad = [(0, 0) for _ in range(len(np.shape(array)) - 1)]
    diff = sample_count - last_dim_len
    npad.append((0, diff))
    return np.pad(array, pad_width=npad, mode="constant", constant_values=pad_value)


def fill_column(
    df: pd.DataFrame,
    col_name: Hashable,
    fill: Union[pd.Series, Mapping, str, int, float],
    na_only: bool = True,
) -> None:
    """
    Fill a column of a DataFrame with the provided values

    Parameters
    ----------
    df
        DataFrame with the column to be filled in
    col_name
        Name of the column to fill (will be created if it doesn't already exist)
    fill
        Values used to fill the series. Acceptable inputs include anything that
        can be used to set the values in a pandas Series
    na_only
        If True, only fill in NaN values (default=True)

    Notes
    -----
    This acts in place on the DataFrame

    """
    if (col_name not in df.columns) or not na_only:
        if col_name in df.columns and df[col_name].notnull().any():
            warnings.warn(f"Overwriting values in {col_name}.")
        df[col_name] = fill
    else:
        df[col_name].fillna(fill, inplace=True)


def df_update(df1: pd.DataFrame, df2: pd.DataFrame, overwrite: bool = True) -> None:
    """
    Performs a DataFrame update that adds new columns to the DataFrame

    This is necessary because pd.DataFrame.update only supports a left join.
    Acts in place on df1.

    Parameters
    ----------
    df1
        Original dataframe
    df2
        Information to update the dataframe with
    overwrite
        Indicates whether to overwrite existing values in the DataFrame

    Returns
    -------
    The updated DataFrame
    """
    # Manually add any new columns
    new_cols = set(df2.columns) - set(df1.columns)
    for col in new_cols:
        df1[col] = np.nan
    # Overwrite existing events
    if overwrite:
        df1.update(df2, overwrite=overwrite)
    else:
        # Deal with weird bug on df.update involving NaN vs NaT
        # Note that this only works -because- overwrite=False
        df2 = df2.reindex_like(df1)
        for col in df2.columns:
            if (col in df1.columns) and (df1.dtypes[col] == "datetime64[ns]"):
                df2[col] = df2[col].astype(df1.dtypes[col])
        df1.update(df2, overwrite=overwrite)


def list_of_str_params(value: Iterable) -> List:
    """
    Make sure a list of params is returned.
    This allows user to specify a single parameter, a list, set, np array, etc.
    to match on.

    Notes
    -----
    Filched from obsplus
    """
    if isinstance(value, str):
        return [value]
    else:
        # try to coerce in a list of str
        try:
            return [str(x) for x in value]
        except TypeError:  # else fallback to str repr
            return [str(value)]


def broadcast_param(
    df: pd.DataFrame,
    param: BroadcastableFloatType,
    col_name: str,
    broadcast_by: Optional[Union[str, Iterable[str]]] = None,
    na_only: bool = True,
):
    """
    Broadcast a parameter to a column in a DataFrame

    Parameters
    ----------
    df
        The dataframe to broadcast the parameter to
    param
        The value(s) to broadcast
    col_name
        The name of the column to broadcast to
    broadcast_by
        The index level(s) to use to perform the broadcast with a dict
    na_only
        If True, only overwrite NaN values
    """

    # Helper functions
    def _retrieve_nested(d, key, default=None):
        if not isinstance(d, dict):
            return d  # Already reached the bottom level, just go with it
        key = list_of_str_params(key)
        lvl = key[0]
        new_d = d.get(lvl, None)
        if new_d is None:
            return  # There isn't anything that matches, going to have to skip it...
        else:
            return _retrieve_nested(new_d, key[1:], default=default)

    def _build_mapping():
        mapping = {}
        for key in broadcast_index.unique():
            val = _retrieve_nested(param, key)
            if val is not None:
                mapping[key] = val
            # else:
            #     warnings.warn(f"No {col_name} matches the index: {key}. Using default value instead.")
        return mapping

    if not len(df):
        raise ValueError("No phases have been added to the StatsGroup")

    if isinstance(param, dict):
        if broadcast_by:
            broadcast_by = list_of_str_params(broadcast_by)
            # Get all relevant part of the multiindex
            broadcast_index = pd.MultiIndex.from_arrays(
                [df.index.get_level_values(x) for x in broadcast_by]
            )
            broadcast_index.names = broadcast_by
            broadcast = _build_mapping()
            param = pd.Series(broadcast_index.map(broadcast))
            param.index = df.index
        else:
            raise TypeError("Dictionary input not supported for specified parameter")
    elif isinstance(param, (Number, pd.Series)):
        pass
    else:
        raise TypeError(f"Unsupported {col_name} format: {type(param)}")
    # Fill in the column
    fill_column(df, col_name=col_name, fill=param, na_only=na_only)
    return df


def inplace(method):
    @functools.wraps(method)
    # Determines whether to modify an object in place or to return a new object
    def if_statement(*args, **kwargs):
        inplace = kwargs.pop("inplace", False)
        self = args[0]
        remainder = args[1:]
        if not inplace:
            self = self.copy()
        out = method(self, *remainder, **kwargs)
        return out

    return if_statement


def _get_alert_function(mode: Literal["warn", "raise", "ignore"], exception=ValueError):
    """
    Return a function which takes a single message and warns, raises, or
    ignores.
    """

    def _warn(msg):
        warnings.warn(msg)

    def _raise(msg):
        raise exception(msg)

    def _ignore(msg):
        pass

    funcs = {"warn": _warn, "raise": _raise, "ignore": _ignore}
    return funcs[mode]


class SourceParameterAggregator:  # This is doing things subtly incorrect (and majorly incorrect for energy...)
    """
    Class for getting event source params from station/phase params.
    """

    # columns which need to be aggregated by median and sum

    _median_aggs = ("moment", "potency", "mw")
    _sum_phase_aggs = ("energy",)

    def __init__(self):
        pass

    def _pivot_phase(
        self, df,
    ):
        """
        Perform pivot to add phase to column names.

        Before the dataframe has index levels phase_hint, event_id.
        The output dataframe has columns {source_param}_{phase_hint}
        and index level of only event_id.
        """
        df_list = []
        phases = np.unique(df.index.get_level_values("phase_hint"))
        for phase in phases:
            df_sub = df.loc[(phase)]
            df_sub.columns = [f"{x}_{phase}" for x in df_sub.columns]
            df_list.append(df_sub)
        out = pd.concat(df_list, axis=1)
        return out

    def aggregate(self, df):
        """
        Perform the aggregations.
        """
        # check we have all but seed_id in index
        assert set(df.index.names) == set(_INDEX_NAMES[:-1])
        # Aggregate each parameter by phase by taking the median
        df_by_phase = self._pivot_phase(df.groupby(["phase_hint", "event_id"]).median())

        # Aggregate appropriate parameters by event by taking the median
        df_by_event = df.groupby("event_id")[self._median_aggs].median()

        # Finally, aggregate by station and event...
        # Group by event and station and filter out stations that don't have both P and S picks
        filtered = df.groupby(["event_id", "seed_id_less"]).filter(lambda x: len(x) > 1)
        # Sum the parameters at each station
        df_by_station = filtered.groupby(["event_id", "seed_id_less"])[
            self._sum_phase_aggs
        ].sum()
        # Then aggregate the parameters by event by taking the median
        df_by_station = df_by_station.groupby("event_id").median()

        # Create one big df and return
        out = pd.concat([df_by_phase, df_by_event, df_by_station], axis=1)
        return out

    __call__ = aggregate
