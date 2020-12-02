"""
Class for keeping track of all metadata used by mopy.
"""
from __future__ import annotations

import warnings
from typing import Optional, Collection, Union, Tuple, Dict, Sequence

import numpy as np
import obsplus
import pandas as pd
from obsplus.utils.events import get_seed_id
from obsplus.utils import get_nslc_series
from obsplus.utils.time import to_utc, to_timedelta64, to_datetime64
from obsplus.utils.geodetics import SpatialCalculator
from obspy import Catalog, Inventory, Stream
from obspy.core import UTCDateTime
from obspy.core.event import Pick, ResourceIdentifier

from mopy.core.base import GroupBase
from mopy.config import get_default_param
from mopy.constants import (
    _INDEX_NAMES,
    NSLC_DTYPES,
    PICK_DTYPES,
    PHASE_WINDOW_DF_DTYPES,
    ChannelPickType,
    AbsoluteTimeWindowType,
)
from mopy.exceptions import DataQualityError, NoPhaseInformationError
from mopy.utils.misc import (
    expand_seed_id,
    _track_method,
    inplace,
    fill_column,
    df_update,
)
from mopy.utils.wrangle import get_phase_window_df


class HomogeneousColumnDescriptor:
    """
    A descriptor for returning values from columns (in the dataframe) which all
    have the same values.
    """

    def __init__(self, column_name):
        self._column_name = column_name

    def __set_name__(self, owner, name):
        self._name = "_" + name

    def __get__(self, instance: StatsGroup, owner):
        # first try to get the data from cache, if emtpy get from df
        if self._name not in instance._cache:
            assert self._column_name in instance.data.columns
            vals = instance.data[self._column_name].unique()
            assert len(vals) == 1, f"{self._column_name} is not homogeneous"
            instance._cache[self._name] = vals[0]
        return instance._cache[self._name]


class _StatsGroup(GroupBase):
    """
    Class for creating information about each channel.

    Parameters
    ----------
    catalog
        Data containing information about the events.
    inventory
        Station data.
    phases
        if Not None, only include phases provided
    """

    processing = ()  # make sure processing attr is present
    sampling_rate = HomogeneousColumnDescriptor("sampling_rate")
    motion_type = HomogeneousColumnDescriptor("motion_type")

    # Customizable methods
    @_track_method
    def add_time_buffer(
        self,
        start: Optional[Union[float, pd.Series]] = None,
        end: Optional[Union[float, pd.Series]] = None,
    ) -> "StatsGroup":
        """
        Method for adding a time before to start and end of windows.

        Returns
        -------
        start
            The time, in seconds, to add to the start of the window
        end
            The time, in seconds, to add to the start of the window
        """
        df = self.data.copy()
        if start is not None:
            df.loc[:, "starttime"] = df["starttime"] - start
        if end is not None:
            df.loc[:, "endtime"] = df["endtime"] + end
        return self.new_from_dict(data=df)


class StatsGroup(_StatsGroup):
    """
    A class for storing metadata about events and channels.
    """

    def __init__(
        self,
        catalog: Catalog,
        inventory: Inventory,
        phases: Optional[Collection[str]] = None,
        restrict_to_arrivals: bool = True,
    ):
        # check inputs
        # st_dict, catalog = self._validate_inputs(catalog, inventory, st_dict)
        catalog = catalog.copy()
        # Convert inventory to a dataframe if it isn't already
        inv_df = obsplus.stations_to_df(inventory)
        inv_df.set_index("seed_id", inplace=True)
        # get a df of all input data, perform sanity checks
        event_station_df = SpatialCalculator()(catalog, inv_df)
        # Calculate hypocentral distance
        event_station_df["hyp_distance_m"] = np.sqrt(
            event_station_df["distance_m"] ** 2
            + event_station_df["vertical_distance_m"] ** 2
        )
        event_station_df.index.names = ["event_id", "seed_id"]
        # we need additional info from the stations, get it and join.
        self.event_station_df = self._join_station_info(inv_df, event_station_df)

        # self._join_station_info()
        df = self._get_meta_df(
            catalog, phases=phases, restrict_to_arrivals=restrict_to_arrivals
        )
        self.data = df
        # st_dict, catalog = self._validate_inputs(catalog, st_dict)
        # # get a df of all input data, perform sanity checks
        # df = self._get_meta_df(catalog, st_dict, phases=phases)
        # self.data = df
        # # add sampling rate to stats
        # self._stats = AttribDict(motion_type=motion_type)
        # init cache
        self._cache = {}

    # Internal methods

    def _validate_inputs(self, events, waveforms) -> Tuple[Dict[str, Stream], Catalog]:
        """
        Asserts some simple checks on inputs. Returns a pruned waveform stream.
        """
        assert isinstance(waveforms, dict)
        # make sure the streams a not empty. create new dict with streams
        # that have some traces and that overlap with ids in events.
        event_ids = {str(eve.resource_id) for eve in events}
        to_keep = {x for x, tr in waveforms.items() if len(tr)}
        st_dict = {x: waveforms[x] for x in to_keep if x in event_ids}
        if not st_dict:
            msg = f"No streams found for events: {events}"
            raise DataQualityError(msg)
        # now create new catalog with data
        cat = events.copy()
        cat.events = [x for x in cat if str(x.resource_id) in st_dict]
        if len(cat) < len(events):
            original_ids = {str(x.resource_id) for x in events}
            filtered_ids = {str(x.resource_id) for x in cat}
            msg = f"missing data for event ids: {original_ids - filtered_ids}"
            warnings.warn(msg)
        return st_dict, cat

    def _get_meta_df(self, catalog, phases=None, restrict_to_arrivals: bool = True):
        """
        Return a dataframe containing pick/duration info.

        Uses defaults, all of which can be overwritten in the config file.

        Columns contain info on time window start/stop as well as traces.
        Index is multilevel using (event_id, phase, seed_id).
        """
        # create a list of sampling rates per channel
        sample_rate = (
            self.event_station_df.reset_index()
            .drop_duplicates("seed_id")
            .set_index(["seed_id_less", "seed_id"])["sample_rate"]
        )
        df_list = []  # list for gathering dataframes
        kwargs = dict(
            dist_df=self.event_station_df,
            phases=phases,
            sample_rate=sample_rate,
            restrict_to_arrivals=restrict_to_arrivals,
        )
        for event in catalog:
            try:
                noise_df, signal_df = self._get_event_meta(event, **kwargs)
            except NoPhaseInformationError:
                dtypes = PHASE_WINDOW_DF_DTYPES.copy()
                dtypes["event_id"] = str
                dtypes["sampling_rate"] = "float64"
                df = pd.DataFrame(columns=dtypes.keys()).astype(dtypes)
                df_list.append(df.set_index(list(_INDEX_NAMES)))
            # except Exception:
            #     warnings.warn(f"failed on {event}")
            else:
                df_list.extend([signal_df, noise_df])
        # concat df and perform sanity checks
        df = pd.concat(df_list, sort=True)
        if not len(df):
            warnings.warn("Catalog did not have any valid picks")
            return df
        df = self._update_meta(df)
        return df

    def _update_meta(self, df: pd.DataFrame) -> pd.DataFrame:
        # get duration/pick dataframes for each event
        # add metadata and validate
        df = self.add_mopy_metadata(df.sort_index())
        self._validate_meta_df(df)
        return df

    def _get_event_meta(
        self, event, dist_df, phases, sample_rate, restrict_to_arrivals: bool
    ):
        """ For an event create dataframes of signal windows and noise windows. """
        event_id = str(event.resource_id)
        # make sure there is one trace per seed_id
        df = self._get_event_phase_window(
            event,
            dist_df,
            sampling_rate=sample_rate,
            restrict_to_arrivals=restrict_to_arrivals,
        )
        # split out noise and non-noise phases
        is_noise = df.phase_hint.str.lower() == "noise"
        noise_df = df[is_noise]
        df = df[~is_noise]
        # if there are specified phases filter out unselected
        if phases:
            df = df[df.phase_hint.isin(phases)]
        # ensure there are no phases with multiple picks on same channel
        assert not df[["phase_hint", "seed_id"]].duplicated().any()
        # add event id to columns
        df["event_id"] = event_id
        # attach traces to new column
        # add sample rates of stream
        df["sampling_rate"] = df["seed_id"].map(sample_rate.droplevel("seed_id_less"))
        assert (df["endtime"] >= df["starttime"]).all(), "negative lengths found!"
        # if any windows have the same start as end time NaN start/end times
        is_zero_len = df["endtime"] == df["starttime"]
        df.loc[is_zero_len, ["endtime", "starttime"]] = np.NaN
        # filter df to only include phases for which data exist, warn
        df_noise = self._get_noise_windows(df, noise_df)
        df = df.set_index(list(_INDEX_NAMES))
        return df_noise, df

    def _join_station_info(self, station_df: pd.DataFrame, event_station_df):
        """ Joins some important station info to the event_station_info dataframe. """
        col_map = dict(
            depth="station_depth",
            azimuth="station_azimuth",
            dip="station_dip",
            sample_rate="sample_rate",
            station="station",
        )
        sta_df = station_df[list(col_map)].rename(columns=col_map)
        event_station_df = event_station_df.join(sta_df).reset_index()
        event_station_df["seed_id_less"] = event_station_df["seed_id"].str[:-1]
        return event_station_df.set_index(
            ["event_id", "seed_id_less", "seed_id"]
        ).sort_index()

    def _get_event_phase_window(
        self, event, dist_df, sampling_rate, restrict_to_arrivals: bool
    ):
        """
        Get the pick time, window start and window end for all phases.
        """
        # determine min duration based on min samples and sec/dist
        # min duration based on required num of samples
        min_samples = get_default_param("min_samples", obj=self)
        min_dur_samps = min_samples / sampling_rate
        # min duration based on distances
        seconds_per_m = get_default_param("seconds_per_meter", obj=self)
        dist = dist_df.loc[str(event.resource_id), "hyp_distance_m"]
        min_dur_dist = pd.Series(dist * seconds_per_m, index=dist.index)
        # the minimum duration is the max the min sample requirement and the
        # min distance requirement
        min_duration = to_timedelta64(np.maximum(min_dur_dist, min_dur_samps))
        # get dataframe
        if not len(event.picks):
            raise NoPhaseInformationError()
        df = get_phase_window_df(
            event,
            min_duration=min_duration,
            channel_codes=set(min_duration.index),
            restrict_to_arrivals=restrict_to_arrivals,
        )  # Todo: should time windows get specified by default,
        # or should they be manually set/attached during the apply defaults method?
        # make sure there are no NaNs
        assert not df.isnull().any().any()
        return df

    def _get_noise_windows(self, phase_df, df):
        """
        Get noise window rows by first looking for noise phase, if None
        just use start of trace.
        """
        # init df for each unique channel that needs a noise spectra
        noise_df = phase_df[~phase_df["seed_id"].duplicated()]
        noise_df["phase_hint"] = "Noise"
        # If no noise spectra is defined use start of traces
        if df.empty:
            # get parameters for determining noise windows start and stop
            noise_end = to_timedelta64(
                get_default_param("noise_end_before_p", obj=self)
            )
            min_noise_dur = to_timedelta64(
                get_default_param("noise_min_duration", obj=self)
            )
            largest_window = (phase_df["endtime"] - phase_df["starttime"]).max()
            min_duration = pd.Series(
                [min_noise_dur, largest_window]
            ).max()  # Necessary to do it this way because max and np.max can't handle NaN/NaT properly
            # set start and stop for noise window
            noise_df["endtime"] = phase_df["starttime"].min() - noise_end
            noise_df["starttime"] = noise_df["endtime"] - min_duration
        else:
            # else use either the noise window defined for a specific station
            # or, if a station has None, use the noise window with the earliest
            # start time
            ser_min = df.loc[df["starttime"].idxmin()]
            t1, t2 = ser_min["starttime"], ser_min["endtime"]
            # drop columns on df and noise df to facilitate merge
            df = df[["network", "station", "starttime", "endtime"]]
            noise_df = noise_df.drop(columns=["starttime", "endtime"])
            noise_df = noise_df.merge(df, how="left", on=["network", "station"])
            # fill nan
            noise_df = noise_df.fillna({"starttime": t1, "endtime": t2})
        # set time between min and max
        noise_df["time"] = (
            noise_df["starttime"] + (noise_df["endtime"] - noise_df["starttime"]) / 2
        )
        # noise_df["time"] = noise_df[["starttime", "endtime"]].mean(axis=1)
        # make sure there are no null values
        out = noise_df.set_index(list(_INDEX_NAMES))
        # drop any duplicate indices
        return out.loc[~out.index.duplicated()]

    def _validate_meta_df(self, df):
        """ Perform simple checks on meta df that should always hold True. """
        # there should be no duplicated indices
        assert not df.index.duplicated().any()
        # all start times should be less than their corresponding end-time
        if df["endtime"].notnull().any():
            subset = df.loc[df["starttime"].notnull() & df["endtime"].notnull()]
            assert (subset["endtime"] > subset["starttime"]).all()
        # ray_paths should all be at least as long as the source-receiver dist
        assert (df["ray_path_length_m"] >= df["hyp_distance_m"]).all()

    def _set_picks_and_windows(
        self, data, mapping, param_name, label, parse_df, set_param
    ):
        """
        Internal method for setting picks and time windows from a dict-like object.
        """
        if isinstance(mapping, str):
            # See if it's a DataFrame
            try:
                mapping = pd.read_csv(mapping)
            except TypeError:
                raise TypeError(f"{param_name} should be a mapping of {label}s")
            parse_df(data, mapping)
        elif isinstance(mapping, pd.DataFrame):
            # It is a DataFrame
            parse_df(data, mapping)
        else:
            raise TypeError(f"Unknown pick input type: {type(mapping).__name__}")
        # TODO: Do we really want to support this? It's inefficient and brittle
        #  If we do, we want to make use of pd.DataFrame.from_dict
        # else:
        #     # Try to run with it
        #     if not hasattr(mapping, "__getitem__"):
        #         raise TypeError(f"{param_name} should be a mapping of {label}s")
        #     for key in mapping:
        #         if key in data.index:
        #             # Index is already in the dataframe, try to overwrite the parameter
        #             warnings.warn(f"Overwriting existing {label}: {key}")
        #             set_param(data, key, mapping[key])
        #         else:
        #             # Index is not in the dataframe, try to attach a new parameter
        #             try:
        #                 self._append_data(data, key, mapping[key], set_param)
        #             except IndexError:
        #                 raise TypeError(f"{param_name} should be a mapping of {label}s")
        return data

    def _append_data(self, data_df, index, params, set_param):
        """ internal method for appending data to StatsGroup.data """
        # make sure the event is in the catalog
        if index[1] not in self.event_station_df.index.get_level_values("event_id"):
            raise KeyError(f"Event is not in the catalog: {index[1]}")

        # make sure the seed id is in the inventory
        if index[2] not in self.event_station_df.index.get_level_values("seed_id"):
            raise KeyError(f"seed_id is not in the inventory: {index[2]}")

        # extract the necessary information from the pick and append it to the data_df
        set_param(data_df, index, params, append=True)

    @staticmethod
    def _make_resource_id(row, data_df):
        """ Internal method to create new resource_ids to attach to StatsGroup.data """
        if row.name in data_df.index:
            return data_df.loc[row.name].pick_id
        else:
            return ResourceIdentifier().id

    def _prep_parse_df(self, df, index, time_cols, data_df):
        df = df.reset_index()
        # Get the station information
        if not set(NSLC_DTYPES.keys()).issubset(df.columns):
            df[list(NSLC_DTYPES.keys())] = expand_seed_id(df["seed_id"])
        # Set the index to what we want
        df["seed_id_less"] = df["seed_id"].str[:-1]
        df = df.set_index(list(index))
        # Convert all of the times to timestamps
        for time in time_cols:
            df[time] = df[time].apply(
                to_datetime64, default=pd.NaT
            )  # , on_none=np.nan)
        # Get the resource_id
        if {"resource_id", "pick_id"}.issubset(df.columns):
            if not df.resource_id == df.pick_id:
                raise KeyError("resource_id and pick_id must be identical")
        elif "resource_id" in df.columns:
            df.rename(columns={"resource_id": "pick_id"}, inplace=True)
        else:
            # resource_ids were not provided, try to make one (but make sure not to overwrite an existing one!)
            df["pick_id"] = df.apply(self._make_resource_id, axis=1, data_df=data_df)
        return df

    # pick-specific
    def _parse_pick_df(
        self, data_df, df
    ):  # This could probably be cleaned up a little bit?
        """ Add a Dataframe of picks to the StatsGroup """
        df = self._prep_parse_df(df, _INDEX_NAMES, ["time"], data_df)
        self._check_unknown_event_station(df)
        # Had to get creative to overwrite the existing dataframe, there may be a cleaner way to do this
        intersect = set(df.columns).intersection(set(data_df.columns))
        diff = set(df.columns).difference(set(data_df.columns))

        # For columns that already exist, update their values
        def _update(row, data_df):
            if row.name in data_df.index:
                warnings.warn(f"Overwriting existing pick: {row.name}")
            data_df.loc[row.name, intersect] = row[intersect]

        df.apply(_update, data_df=data_df, axis=1)

        # Create new columns for the ones that don't exist
        for col in diff:
            data_df.loc[df.index, col] = df[col]

    def _check_unknown_event_station(self, df):
        """ Raise if the event/station pair aren't in the list """
        if not set(df.droplevel("phase_hint").index).issubset(
            self.event_station_df.index
        ):
            diff = set(df.droplevel("phase_hint").index).issubset(
                self.event_station_df.index
            )
            raise KeyError(f"Event/station pair(s) does not exist: {diff}")

    # Methods for adding data, material properties, and other coefficients
    @inplace
    def set_picks(
        self, picks: ChannelPickType, inplace=False
    ) -> Optional["StatsGroup"]:
        """
        Method for adding picks to the ChannelInfo

        Parameters
        ----------
        picks
            Mapping containing information about picks, their phase type, and
            their associated metadata. The mapping can be a pandas DataFrame
            containing the following columns: [event_id, seed_id, phase, time],
            or a dictionary of the form
            {(phase, event_id, seed_id): obspy.UTCDateTime or obspy.Pick}.
        inplace
            If True ChannelInfo will be modified inplace, else return a copy.
        """
        self._set_picks_and_windows(
            self.data, picks, "picks", "pick", self._parse_pick_df, self._set_pick
        )
        self.data = self._update_meta(self.data)
        return self

    @inplace
    def set_abs_time_windows(
        self, time_windows: AbsoluteTimeWindowType
    ) -> Optional["StatsGroup"]:
        """
        Method for applying absolute time windows to the ChannelInfo

        Parameters
        ----------
        time_windows
            Mapping containing start and end times for pick time windows. The mapping can be a pandas DataFrame containing
            the following columns: [event_id, seed_id, phase, starttime, endtime], or a dictionary of the form
            {(phase, event_id, seed_id): (starttime, endtime)}.

        Other Parameters
        ----------------
        inplace
            Flag indicating whether the ChannelInfo should be modified inplace or a new copy should be returned
        """
        self._set_picks_and_windows(
            self.data,
            time_windows,
            "time_windows",
            "time window",
            self._parse_tw_df,
            self._set_tw,
        )
        self.data = self._update_meta(self.data)
        return self

    @inplace
    def set_rel_time_windows(self, **time_windows) -> Optional["StatsGroup"]:
        """
        Method for applying relative time windows to the ChannelInfo

        Parameters
        ----------
        time_windows
            The time windows are set on a per-phase basis for arbitrary phase types through the following format:
            phase=(before_pick, after_pick). For example, P=(0.1, 1), S=(0.5, 2), Noise=(0, 5). Note that phase names
            are limited to valid attribute names (alphanumeric, cannot start with a number).

        Other Parameters
        ----------------
        inplace
            Flag indicating whether the ChannelInfo should be modified inplace or a new copy should be returned
        """
        # TODO: I'm going to gloss over this for right now because it doesn't affect my use case, but this might be overwriting user-provided start and end times?
        # Loop over each of the provided phase
        for ph, tw in time_windows.items():
            if not isinstance(tw, Sequence) or isinstance(tw, str):
                raise TypeError(
                    f"time windows must be a tuples of start and end times: {ph}"
                )
            if not len(tw) == 2:
                raise ValueError(f"time windows must be a tuple of floats: {ph}={tw}")
            # Get all of the picks that have a matching phase
            phase_ind = self.data.index.get_level_values("phase_hint") == ph
            # If none of the picks match, issue a warning and move on
            if not phase_ind.any():
                warnings.warn(f"No picks matching phase type: {ph}")
                continue
            if (tw[0] + tw[1]) < 0:
                raise ValueError(f"Time after must occur after time before: {ph}")
            time_before = to_timedelta64(tw[0])
            time_after = to_timedelta64(tw[1])
            # Otherwise, set the time windows
            if (
                self.data.loc[phase_ind, "starttime"].notnull().any()
                or self.data.loc[phase_ind, "endtime"].notnull().any()
            ):
                warnings.warn(
                    "Overwriting existing time windows for one or more phases."
                )
            self.data.loc[phase_ind, "starttime"] = (
                self.data.loc[phase_ind, "time"] - time_before
            )
            self.data.loc[phase_ind, "endtime"] = (
                self.data.loc[phase_ind, "time"] + time_after
            )
        self.data = self._update_meta(self.data)
        return self

    def apply_defaults(self, inplace: bool = False):
        """
        Method to populate any empty StatsGroup parameters with defaults.

        Parameters
        ----------
        inplace
            If True ChannelInfo will be modified inplace, else return a copy.
        """
        df = self.add_defaults(self.data, na_only=True)
        self._validate_meta_df(df)
        return self.new_from_dict(data=df, inplace=inplace)

    def _set_pick(self, data_df, index, pick_info, append=False):
        """ write the pick information to the dataframe """
        if isinstance(pick_info, Pick):
            # parse a pick object
            for col in PICK_DTYPES:
                if col == "time":
                    data_df.loc[index, "time"] = pick_info.time.timestamp
                elif col == "pick_id":
                    data_df.loc[index, "pick_id"] = pick_info.resource_id.id
                else:
                    data_df.loc[index, col] = pick_info.__dict__[col]
            data_df.loc[index, "phase_hint"] = pick_info.__dict__["phase_hint"]
            data_df.loc[index, list(NSLC_DTYPES.keys())] = list(
                get_seed_id(pick_info).split(".")
            )
        else:
            # assign the provided pick time to the dataframe
            try:
                time = UTCDateTime(pick_info).timestamp
            except TypeError:
                raise TypeError("Pick time must be an obspy UTCDateTime")
            else:
                data_df.loc[index, "time"] = time
            # Do the nslc info
            data_df.loc[index, list(NSLC_DTYPES.keys())] = list(
                index[-1].split(".")
            )  # seed_id
            if append:
                # Since there is no resource_id for the pick, create a new one
                data_df.loc[index, "pick_id"] = ResourceIdentifier().id

    # time-window-specific
    def _parse_tw_df(self, data_df, df):
        """ Add a Dataframe of time_windows to the StatsGroup """
        df = self._prep_parse_df(df, _INDEX_NAMES, ["starttime", "endtime"], data_df)
        self._check_unknown_event_station(df)
        if not (df["endtime"] > df["starttime"]).all():
            raise ValueError("time window starttime must be earlier than endtime")

        # Update the data_df
        intersect = set(df.columns).intersection(set(data_df.columns))

        def _update(row, data_df):
            if (row.name in data_df.index) and pd.notnull(
                data_df.loc[row.name, "starttime"]
            ):
                warnings.warn(f"Overwriting existing time_window: {row.name}")
            elif row.name not in data_df.index:
                data_df.loc[row.name, "time"] = row.starttime
            data_df.loc[row.name, intersect] = row[intersect]

        df.apply(_update, data_df=data_df, axis=1)

    def _set_tw(self, data_df, index, pick_info, append=False):
        """ write the time window to the dataframe """
        # Get the start and end times
        try:
            starttime = UTCDateTime(pick_info[0]).timestamp
        except TypeError:
            raise TypeError("starttime must be an obspy UTCDateTime")
        try:
            endtime = UTCDateTime(pick_info[1]).timestamp
        except TypeError:
            raise TypeError("endtime must be an obspy UTCDateTime")
        if endtime <= starttime:
            raise ValueError("time window starttime must be earlier than endtime")

        # Set the start and end times
        data_df.loc[index, "starttime"] = starttime
        data_df.loc[index, "endtime"] = endtime

        if append:
            # Populate the minimum information for it to be a valid pick
            data_df.loc[index, list(NSLC_DTYPES.keys())] = list(index[-1].split("."))
            data_df.loc[index, ["time", "pick_id"]] = [
                starttime,
                ResourceIdentifier().id,
            ]

    def _append_tw(self, data_df, index, pick_info):
        # make sure the event is in the catalog
        if index[1] not in self.event_station_df.index.levels[0]:
            raise KeyError(f"Event is not in the catalog: {index[1]}")

        # make sure the seed id is in the inventory
        if index[2] not in self.event_station_df.index.levels[1]:
            raise KeyError(f"seed_id is not in the inventory: {index[2]}")

        # extract the necessary information from the pick and append it to the data_df
        self._set_pick(data_df, index, pick_info, append=True)

    # Customizable methods
    def add_time_buffer(
        self,
        start: Optional[Union[float, pd.Series]] = None,
        end: Optional[Union[float, pd.Series]] = None,
    ) -> "ChannelInfo":
        """
        Method for adding a time before to start and end of windows.

        Returns
        -------
        start
            The time, in seconds, to add to the start of the window
        end
            The time, in seconds, to add to the start of the window
        """
        df = self.data.copy()
        if start is not None:
            df.loc[:, "starttime"] = df["starttime"] - to_timedelta64(start)
        if end is not None:
            df.loc[:, "endtime"] = df["endtime"] + to_timedelta64(end)
        return self.new_from_dict(data=df)

    def add_mopy_metadata(self, df: pd.DataFrame):
        """
        Responsible for adding any needed metadata to df.
        """
        # TODO: I have a couple of concerns/comments here that need to be looked at more closely:
        #  1. Is the source reciever distance only in plan view, or is it a hypocentral distance? The SpatialCalculator in obsplus returns plan-view distance, I believe...
        #  2. Could things be named more unambiguously to clear this up for the future?
        #  3. If this is just the plan view distance, then the ray path length is probably incorrect and needs to be double-checked
        # add source-receiver distance
        df = self.add_source_receiver_distance(df)
        # add ray_path_lengths
        df = self.add_ray_path_length(
            df
        )  # How is this different from source-receiver distance (one is straight line, the other can be arbitrary)

        # add travel time
        df = self.add_travel_time(df)
        return df

    def add_defaults(
        self, df: pd.DataFrame, na_only: bool = True
    ):  # TODO: There is a problem with this where it has the ability to override non-nan values....
        """
        Populate nan values in df with default values
        """
        # add velocity
        df = self.add_source_velocity(df, na_only=na_only)
        # add radiation pattern corrections
        df = self.add_radiation_coeficient(df, na_only=na_only)
        # add quality factors
        df = self.add_quality_factor(df, na_only=na_only)
        # add geometric spreading factor
        df = self.add_spreading_coefficient(df, na_only=na_only)
        # add density
        df = self.add_density(df, na_only=na_only)
        # add shear modulus
        df = self.add_shear_modulus(df, na_only=na_only)
        # add free surface correction
        df = self.add_free_surface_coefficient(df, na_only=na_only)
        return df

    def add_source_receiver_distance(self, df: pd.DataFrame):
        """
        Add (hypocentral) source-receiver distance to each pick.
        """
        dist = self.event_station_df.copy()
        dist_cols = ["distance_m", "vertical_distance_m", "hyp_distance_m", "azimuth"]
        for col in dist_cols:
            if col not in df.columns:
                df[col] = np.nan
        # dist.index.names = ("event_id", "seed_id")
        # there need to be common index names
        assert set(dist.index.names).issubset(set(df.index.names))
        # broadcast distance df to shape of df and join
        phases = list(df.index.get_level_values("phase_hint").unique().values)
        distdf = pd.concat(
            [dist for _ in range(len(phases))], keys=phases, names=["phase_hint"]
        )
        df = df.copy()
        df_update(df, distdf, overwrite=False)
        return df

    def add_ray_path_length(self, df, ray_length=None):
        """
        Add ray path distance to each channel event pair.

        By default only a simple straight-ray distance is used. This can
        be overwritten to use a more accurate ray-path distance.
        """
        ray_length = (
            df["hyp_distance_m"] if ray_length is not None else df["hyp_distance_m"]
        )
        df["ray_path_length_m"] = ray_length
        return df

    def add_source_velocity(
        self, df: pd.DataFrame, velocity: Optional[float] = None, na_only: bool = True
    ):
        """ Add the velocity to meta dataframe """
        # Determine what the appropriate value should be
        if velocity is None:
            vel_map = dict(
                S=get_default_param("s_velocity"), P=get_default_param("p_velocity")
            )
            velocity = pd.Series(df.index.get_level_values("phase_hint").map(vel_map))
            velocity.index = df.index
        # Fill in the column
        fill_column(df, col_name="source_velocity", fill=velocity, na_only=na_only)
        return df

    def add_spreading_coefficient(
        self, df: pd.DataFrame, spreading: Optional[float] = None, na_only: bool = True
    ):
        """
        Add the spreading coefficient. If None assume spreading 1 / r.
        """
        # Determine what the appropriate value should be
        # TODO: I actually think this should be based on hypocentrol distance, not ray path length, but I'm still trying to justify it to myself
        spreading = 1 / df["ray_path_length_m"] if spreading is None else spreading
        # Fill the column
        fill_column(
            df, col_name="spreading_coefficient", fill=spreading, na_only=na_only
        )
        return df

    def add_radiation_coeficient(
        self,
        df: pd.DataFrame,
        radiation_coefficient: Optional[float] = None,
        na_only=True,
    ):
        """ Add the factor used to correct for radiation pattern. """
        if radiation_coefficient is None:
            radiation_coef_map = dict(
                S=get_default_param("s_radiation_coefficient"),
                P=get_default_param("p_radiation_coefficient"),
                Noise=1.0,
            )
            radiation_coefficient = pd.Series(
                df.index.get_level_values("phase_hint").map(radiation_coef_map)
            )
            radiation_coefficient.index = df.index
        fill_column(
            df,
            col_name="radiation_coefficient",
            fill=radiation_coefficient,
            na_only=na_only,
        )
        return df

    def add_quality_factor(
        self,
        df: pd.DataFrame,
        quality_factor: Optional[float] = None,
        na_only: bool = True,
    ):
        """ Add the quality factor """
        if quality_factor is None:
            quality_factor = get_default_param("quality_factor")
        fill_column(df, col_name="quality_factor", fill=quality_factor, na_only=na_only)
        return df

    def add_density(
        self, df: pd.DataFrame, density: Optional[float] = None, na_only: bool = True
    ):
        """
        Add density to the meta dataframe. If None, use defaults.
        """
        if density is None:
            density = get_default_param("density")
        fill_column(df, col_name="density", fill=density, na_only=na_only)
        return df

    def add_shear_modulus(
        self,
        df: pd.DataFrame,
        shear_modulus: Optional[float] = None,
        na_only: bool = True,
    ):
        """
        Add the shear modulus to the meta dataframe
        """
        if shear_modulus is None:
            shear_modulus = get_default_param("shear_modulus")
        fill_column(df, col_name="shear_modulus", fill=shear_modulus, na_only=na_only)
        return df

    def add_free_surface_coefficient(
        self,
        df: pd.DataFrame,
        free_surface_coefficient: float = None,
        na_only: bool = True,
    ):
        """
        Add the coefficient which corrects for free surface.

        By default just uses 1/2 if the depth of the instrument is 0, else 1.
        """
        if free_surface_coefficient is None:
            free_surface_map = (
                self.event_station_df.reset_index()[["seed_id", "station_depth"]]
                .drop_duplicates(subset="seed_id")
                .set_index("seed_id")
            )
            free_surface_map["free_surface_coefficient"] = free_surface_map[
                "station_depth"
            ].apply(lambda x: 2.0 if np.isclose(x, 0.0) else 1.0)
            free_surface_map = free_surface_map["free_surface_coefficient"].to_dict()
            free_surface_coefficient = pd.Series(
                df.index.get_level_values("seed_id").map(free_surface_map)
            )
            free_surface_coefficient.index = df.index
        fill_column(
            df,
            col_name="free_surface_coefficient",
            fill=free_surface_coefficient,
            na_only=na_only,
        )
        return df

    def add_travel_time(self, df):
        """
        Add the travel time of each phase to the dataframe.
        """
        return df
