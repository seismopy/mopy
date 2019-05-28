import copy
import pickle
from pathlib import Path
from typing import TypeVar, Type, List, Tuple

import pandas as pd
from obsplus.constants import NSLC

from mopy.core import StatsGroup
from mopy.utils import _source_process, new_from_dict

DFG = TypeVar("DFG", bound="DataFrameGroupBase")


class DataFrameGroupBase:
    """ Base class for TraceGroup and SpectrumGroup. """

    _channel_info: StatsGroup
    # stats: AttribDict
    data: pd.DataFrame
    processing: List[Tuple[str, str]]

    @property
    def stats(self):
        return self._channel_info.data

    @stats.setter
    def stats(self, item):
        self._channel_info.data = item

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
        proc = getattr(self._stats_group, "processing", ())
        return any(name in x for x in proc)

    def expand_seed_id(self: DFG) -> DFG:
        """
        Expand the seed_id to include network, station, location, and channel.

        This is useful, for example, to groupby station.
        """
        df_old = self.data
        meta_old = self.meta
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

    new_from_dict = new_from_dict

    def copy(self):
        """ Perform a deep copy. """
        return copy.deepcopy(self)

    @_source_process
    def abs(self: DFG) -> DFG:
        """
        Take the absolute value of all values in dataframe.
        """
        return self.new_from_dict({"data": abs(self.data)})

    @_source_process
    def add(self: DFG, other: DFG) -> DFG:
        """
        Add two source_groupy things together.
        """
        return self.new_from_dict(dict(data=self.data + other))

    @_source_process
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

    def __str__(self):
        events = self.data.index.get_level_values("event_id").unique()
        msg = f"SourceGroup with {len(events)} Events"
        return msg

    __repr__ = __str__
