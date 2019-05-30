"""
Base class for mopy's dataframe based classes which have a StatsGroup.
"""
from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import TypeVar, Type

import pandas as pd
from obsplus.constants import NSLC
from obsplus.utils import iterate

import mopy
from mopy.utils import _track_method

DFG = TypeVar("DFG", bound="DataFrameGroupBase")


class ProxyAttribute:
    """ A simple descriptor to treat an attribute as a proxy. """

    def __init__(self, attr_name, object_name="stats_group"):
        self.obj_name = object_name
        self.attr_name = attr_name

    def __get__(self, instance, owner):
        obj = getattr(instance, self.obj_name)
        return getattr(obj, self.attr_name)

    def __set__(self, instance, value):
        obj = getattr(instance, self.obj_name)
        setattr(obj, self.attr_name, value)


class GroupBase:
    """ Base class for TraceGroup and SpectrumGroup. """

    # stats: AttribDict
    data: pd.DataFrame

    def to_pickle(self, path=None):
        """ Save the object to pickle format. """
        byt = pickle.dumps(self)
        if path is not None:
            path.parent.mkdir(exist_ok=True, parents=True)
            path = Path(path) if not hasattr(path, "open") else path
            with path.open("wb") as fi:
                pickle.dump(self, fi, protocol=pickle.HIGHEST_PROTOCOL)
        return byt

    @classmethod
    def from_pickle(cls: Type[DFG], path) -> DFG:
        """ Read a source group from a pickle. """
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

    def expand_seed_id(self: DFG) -> DFG:
        """
        Expand the seed_id to include network, station, location, and channel.

        This is useful, for example, to groupby station.
        """
        # TODO finsih this
        df_old = self.data
        index = self._get_expanded_index()
        df = pd.DataFrame(df_old.values, columns=df_old.columns, index=index)
        # metat = 1
        return self.new_from_dict({"data": df})

    def collapse_seed_id(self: DFG) -> DFG:
        """
        Collapse the network, station, location, channel back to seed_id.
        """
        return self
        ind = self.data.index

    def _get_expanded_index(self) -> pd.Index:
        """ return an expanded index. """
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
        """ collapse and index that has """
        pass

    def new_from_dict(self: DFG, update: dict) -> DFG:
        """
        Create a new object from a dict input to the old object.
        """
        copy = self.copy()
        copy.__dict__.update(update)
        return copy

    def get_column_or_index(self, name: str) -> pd.Series:
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

    def assert_has_column_or_index(self, name):
        """
        Assert that the object has an index or column.

        Parameters
        ----------
        name

        Raises
        ------
        ValueError if the dataframe does not have column or index name.
        """
        try:
            self.get_column_or_index(name)
        except KeyError as e:
            raise e

    def copy(self):
        """ Perform a deep copy. """
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
        return self.new_from_dict({"data": df})

    @property
    def index(self):
        """
        Return the index of the dataframe.
        """
        return self.data.index

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
        """ set the stats group """
        self.stats_group = stats_group.copy()

    @_track_method
    def abs(self: DFG) -> DFG:
        """
        Take the absolute value of all values in dataframe.
        """
        return self.new_from_dict({"data": abs(self.data)})

    @_track_method
    def add(self: DFG, other: DFG) -> DFG:
        """
        Add two source_groupy things together.
        """
        return self.new_from_dict(dict(data=self.data + other))

    @_track_method
    def multiply(self: DFG, other: DFG) -> DFG:
        """
        Multiply two source-groupy things.
        """
        return self.new_from_dict(dict(self.data * other))

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
