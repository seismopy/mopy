"""
Base class for mopy's dataframe based classes which have a StatsGroup.
"""
from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import TypeVar, Type, Optional

import numpy as np
import pandas as pd
import scipy.signal
from obsplus.constants import NSLC
from obsplus.utils import iterate

import mopy
from mopy.utils.misc import _track_method

DFG = TypeVar("DFG", bound="DataFrameGroupBase")


class ProxyAttribute:
    """A simple descriptor to treat an attribute as a proxy."""

    def __init__(self, attr_name, proxy_name="stats_group"):
        self.proxy_name = proxy_name
        self.attr_name = attr_name

    def __get__(self, instance, owner):
        # breakpoint()
        obj = getattr(instance, self.proxy_name)
        return getattr(obj, self.attr_name)

    def __set__(self, instance, value):
        obj = getattr(instance, self.proxy_name)
        setattr(obj, self.attr_name, value)


class GroupBase:
    """Base class for TraceGroup and SpectrumGroup."""

    # stats: AttribDict
    data: pd.DataFrame

    def to_pickle(self, path=None):
        """Save the object to pickle format."""
        byt = pickle.dumps(self)
        if path is not None:
            path = Path(path)
            path.parent.mkdir(exist_ok=True, parents=True)
            path = Path(path) if not hasattr(path, "open") else path
            with path.open("wb") as fi:
                pickle.dump(self, fi, protocol=pickle.HIGHEST_PROTOCOL)
        return byt

    @classmethod
    def from_pickle(cls: Type[DFG], path) -> DFG:
        """Read a source group from a pickle."""
        if isinstance(path, bytes):
            return pickle.loads(path)
        path = path if hasattr(path, "open") else Path(path)
        with path.open("rb") as fi:
            return pickle.load(fi)

    def in_processing(self, name):
        """
        Return True in name is a substring of any of the processing strings.
        """
        proc = getattr(self.stats_group, "processing", ())
        return any(name in x for x in proc)

    def _get_expanded_index(self) -> pd.Index:
        """return an expanded index."""
        # expand seed id
        old_index = self.data.index
        seed_id = old_index.get_level_values("seed_id").to_series()
        nslc = seed_id.str.split(".", expand=True)
        nslc.columns = list(NSLC)
        # add old index values to keep back
        nslc["phase_hint"] = old_index.get_level_values("phase_hint").values
        nslc["event_id"] = old_index.get_level_values("event_id").values
        cols = ["phase_hint", "event_id", "network", "station", "location", "channel"]
        return pd.MultiIndex.from_arrays(nslc[cols].values.T, names=cols)

    def _get_collapsed_index(self) -> pd.Index:
        """collapse and index that has"""
        pass

    def new_from_dict(self: DFG, *, inplace=False, **kwargs) -> DFG:
        """
        Create a new object from a dict input to the old object.
        """
        # if acting in-place just update instance state and return
        if inplace:
            self.__dict__.update(kwargs)
            return self
        # else make new instance and copy only what is not in kwargs
        # then return new instance
        new = self.__new__(self.__class__)
        # get dict of attrs to be copied
        new_dict = {
            i: copy.deepcopy(v)
            for i, v in self.__dict__.items()
            if i not in kwargs and i != "_cache"
        }
        new_dict["_cache"] = {}  # reset cache
        new.__dict__.update(new_dict)
        new.__dict__.update(kwargs)
        return new

    def get_column(self, name: str) -> pd.Series:
        """
        Return a Series of values from a dataframe column or index values.

        Parameters
        ----------
        name
            The name of the column (or index level) to return.
        """
        cols = self.data.columns
        index = self.data.index
        if name in cols:
            return self.data[name]
        elif name in set(iterate(getattr(index, "names", "name"))):
            vals = index.get_level_values(name)
            return pd.Series(vals, index=index)
        else:
            msg = f"{name} is not a column or index level"
            raise KeyError(msg)

    def assert_column(self, name, raise_on_null: Optional[str] = None):
        """
        Assert that the object has an index or column.

        Parameters
        ----------
        name
            The name of the column or index value to return.
        raise_on_null
            Either None - do nothing, 'any' - raise Value Error if any
            nulls are found, or 'all'- raise ValueError if all nulls are found.

        Raises
        ------
        KeyError
            If the dataframe does not have column or index name.
        ValueError
            If raise_on_null is used and required conditions are not met.
        """
        try:
            col = self.get_column(name)
        except KeyError as e:
            raise e
        if raise_on_null:
            assert raise_on_null in {"all", "any"}
            isnull = pd.isnull(col)
            if getattr(isnull, raise_on_null)():
                msg = f"column '{name}' has {raise_on_null} null values."
                raise ValueError(msg)

    def copy(self):
        """Perform a deep copy."""
        return copy.deepcopy(self)

    def add_columns(self, **kwargs):
        """
        Add any number of columns to the containing dataframe.

        Parameters
        ----------
        kwargs
            Used to specify column names.
        """
        df = self.data.copy()
        for item, value in kwargs.items():
            df[item] = value
        return self.new_from_dict(data=df)

    def dropna(self, axis=0, how="any", thresh=None, subset=None):
        """
        Return object with labels on given axis omitted where data are missing.

        Notes
        -----
        See pandas.DataFrame.dropna for parameter meaning.
        """
        df = self.data.dropna(axis=axis, how=how, thresh=thresh, subset=subset)
        return self.new_from_dict(data=df)

    @property
    def index(self):
        """
        Return the index of the dataframe.
        """
        return self.data.index

    def __len__(self):
        return len(self.data)

    def __str__(self):
        cls_name = self.__class__.__name__
        cols = self.data.columns
        msg = f"{cls_name} with {len(self.data)} rows and columns:\n{cols}"
        return msg

    __repr__ = __str__


class DataGroupBase(GroupBase):
    """
    Base class for Group objects that contain Data (not just meta data),
    eg TraceGroup and SpectrumGroup.
    """

    stats_group: mopy.core.StatsGroup

    # proxy attributes
    stats = ProxyAttribute("data")
    processing = ProxyAttribute("processing")
    sampling_rate = ProxyAttribute("sampling_rate")
    motion_type = ProxyAttribute("motion_type")

    def __init__(self, stats_group):
        """set the stats group"""
        self.stats_group = stats_group.copy()

    @_track_method
    def abs(self: DFG) -> DFG:
        """
        Take the absolute value of all values in dataframe.
        """
        return self.new_from_dict(data=abs(self.data))

    @_track_method
    def add(self: DFG, other: DFG) -> DFG:
        """
        Add two source_groupy things together.
        """
        return self.new_from_dict(data=self.data + other)

    @_track_method
    def multiply(self: DFG, other: DFG) -> DFG:
        """
        Multiply two source-groupy things.
        """
        return self.new_from_dict(data=self.data * other)

    @_track_method
    def detrend(self: DFG, type="linear") -> DFG:
        """
        Detrend the data, accounting for NaN values.

        Parameters
        ----------
        type
            The type of detrend. The following are supported:
                "linear" - fit a line to all non-NaN values and subract it.
                "constant" - simple subract the mean of all non-NaN values.
        """
        assert type in {"linear", "constant"}
        df = self.data

        if type == "constant":
            mean = df.mean(axis=1)
            out = df.subtract(mean, axis=0)
            return self.new_from_dict(data=out)
        elif type == "linear":
            values = np.copy(df.values)
            not_nan = ~np.isnan(values)
            # if everything is not NaN used fast path
            if np.all(not_nan):
                values = scipy.signal.detrend(values, axis=1)
            else:  # complicated logic to handle NaN
                vals = values[not_nan]  # flatten to only include non NaN
                # get indices of NoN Nan
                row_ind = np.indices(df.values.shape)[0][not_nan]
                bpoints = np.where(row_ind[:-1] < row_ind[1:])[0] + 1
                # apply piece-wise detrend
                detrended = scipy.signal.detrend(vals, type=type, bp=bpoints)
                # create array of start_ind, stop_ind of detrended
                ind_end = np.concatenate([bpoints, [len(vals)]])
                ind_start = np.zeros_like(ind_end)
                ind_start[1:] = ind_end[:-1]
                # add detrended data back to array
                for row_num, (ind1, ind2) in enumerate(zip(ind_start, ind_end)):
                    ar = detrended[ind1:ind2]
                    values[row_num, 0 : len(ar)] = ar
            # create new df and return
            df = pd.DataFrame(values, index=df.index, columns=df.columns)
            return self.new_from_dict(data=df)

    def copy(self):
        """Perform a deep copy."""
        cp = super().copy()
        # Make sure the stats_group is also deep copied
        cp.stats_group = copy.deepcopy(cp.stats_group)
        return cp

    def __abs__(self):
        return self.abs()

    def __add__(self, other):
        return self.add(other)

    def __iadd__(self, other):
        return self.add(other)

    def __mul__(self, other):
        return self.multiply(other)

    def __imul__(self, other):
        return self.multiply(other)

    def __truediv__(self, other):
        return self.new_from_dict(data=self.data / other)

    def __floordiv__(self, other):
        return self.new_from_dict(data=self.data // other)
