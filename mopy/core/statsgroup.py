"""
Class for keeping track of all metadata used by mopy.
"""
from __future__ import annotations

import warnings
from typing import Optional, Collection, Union, Tuple, Dict

import numpy as np
import obsplus
import pandas as pd
from obsplus.events.utils import get_seed_id
from obsplus.utils import get_distance_df, get_nslc_series, to_timestamp
from obspy import Catalog, Inventory, Stream
from obspy.core import UTCDateTime
from obspy.core.event import Pick, ResourceIdentifier

from mopy.core.base import GroupBase
from mopy.config import get_default_param
from mopy.constants import (
    _INDEX_NAMES,
    STAT_COLS,
    NSLC,
    PICK_COLS,
    ChannelPickType,
    AbsoluteTimeWindowType,
)
from mopy.exceptions import DataQualityError, NoPhaseInformationError
from mopy.utils import get_phase_window_df, expand_seed_id, _track_method, inplace


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
        return self.new_from_dict({"data": df})


class StatsGroup(_StatsGroup):
    """
    A class for storing metadata about events and channels.
    """

    def __init__(
        self,
        catalog: Catalog,
        inventory: Inventory,
        phases: Optional[Collection[str]] = None,
    ):
        # check inputs
        # st_dict, catalog = self._validate_inputs(catalog, inventory, st_dict)
        catalog = catalog.copy()
        # get a df of all input data, perform sanity checks
        self.event_station_info = get_distance_df(catalog, inventory)
        self.event_station_info.index.names = ["event_id", "seed_id"]
        self._join_station_info(obsplus.stations_to_df(inventory))
        df = self._get_meta_df(catalog, inventory, phases=phases)
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

    def _get_meta_df(self, catalog, inventory, phases=None):
        """
        Return a dataframe containing pick/duration info.

        Uses defaults, all of which can be overwritten in the config file.

        Columns contain info on time window start/stop as well as traces.
        Index is multilevel using (event_id, phase, seed_id).
        """
        # calculate source-receiver distance.
        dist_df = self.event_station_info
        sta_df = obsplus.stations_to_df(inventory)
        # create a list of sampling rates per channel
        sample_rate = sta_df.set_index(get_nslc_series(sta_df))[
            "sample_rate"
        ]  # Should add this to event_station_info?
        df_list = []  # list for gathering dataframes
        kwargs = dict(dist_df=dist_df, phases=phases, sample_rate=sample_rate)
        for event in catalog:
            try:
                noise_df, signal_df = self._get_event_meta(event, **kwargs)
            except NoPhaseInformationError:
                df = pd.DataFrame(
                    columns=STAT_COLS + _INDEX_NAMES
                )  # , dtype=CHAN_DTYPES)
                return df.set_index(list(_INDEX_NAMES))
            except Exception:
                warnings.warn(f"failed on {event}")
            else:
                df_list.extend([signal_df, noise_df])
        # concat df and perform sanity checks
        df = pd.concat(df_list, sort=True)
        df = self._update_meta(df)
        return df

    def _update_meta(self, df: pd.DataFrame) -> pd.DataFrame:
        # get duration/pick dataframes for each event
        # add metadata and validate
        df = self.add_mopy_metadata(df.sort_index())
        self._validate_meta_df(df)
        return df

    def _get_event_meta(self, event, dist_df, phases, sample_rate):
        """ For an event create dataframes of signal windows and noise windows. """
        event_id = str(event.resource_id)
        # make sure there is one trace per seed_id
        df = self._get_event_phase_window(event, dist_df, sampling_rate=sample_rate)
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
        df["sampling_rate"] = df["seed_id"].map(sample_rate)
        assert (df["endtime"] >= df["starttime"]).all(), "negative lengths found!"
        # if any windows have the same start as end time NaN start/end times
        is_zero_len = df["endtime"] == df["starttime"]
        df.loc[is_zero_len, ["endtime", "starttime"]] = np.NaN
        # filter df to only include phases for which data exist, warn
        df_noise = self._get_noise_windows(df, noise_df)
        df = df.set_index(list(_INDEX_NAMES))
        return df_noise, df

    def _join_station_info(self, station_df: pd.DataFrame):
        """ Joins some important station info to the event_station_info dataframe. """
        col_map = dict(
            depth="station_depth", azimuth="station_azimuth", dip="station_dip"
        )
        sta_df = station_df.set_index("seed_id")[list(col_map)]
        # There has to be a more efficient pandas way to do this, but I'm just not having any luck at the moment
        for num, row in sta_df.iterrows():
            self.event_station_info.loc[(slice(None), num), "station_depth"] = row.depth
            self.event_station_info.loc[
                (slice(None), num), "station_azimuth"
            ] = row.azimuth
            self.event_station_info.loc[(slice(None), num), "station_dip"] = row.dip
        return

    def _get_event_phase_window(self, event, dist_df, sampling_rate):
        """
        Get the pick time, window start and window end for all phases.
        """
        # determine min duration based on min samples and sec/dist
        # min duration based on required num of samples
        min_samples = get_default_param("min_samples", obj=self)
        min_dur_samps = min_samples / sampling_rate
        # min duration based on distances
        seconds_per_m = get_default_param("seconds_per_meter", obj=self)
        dist = dist_df.loc[str(event.resource_id), "distance"]
        min_dur_dist = pd.Series(dist * seconds_per_m, index=dist.index)
        # the minimum duration is the max the min sample requirement and the
        # min distance requirement
        min_duration = np.maximum(min_dur_dist, min_dur_samps)
        # get dataframe
        if not len(event.picks):
            raise NoPhaseInformationError()
        df = get_phase_window_df(
            event, min_duration=min_duration, channel_codes=set(min_duration.index)
        )  # Todo: should time windows get specified by default, or should they be manually set/attached during the apply defaults method?
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
            endtime = get_default_param("noise_end_before_p", obj=self)
            min_noise_dur = get_default_param("noise_min_duration", obj=self)
            largest_window = (phase_df["endtime"] - phase_df["starttime"]).max()
            min_duration = max(min_noise_dur, largest_window)
            # set start and stop for noise window
            noise_df["endtime"] = phase_df["starttime"].min() - endtime
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
        noise_df["time"] = noise_df[["starttime", "endtime"]].mean(axis=1)
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
        assert (df["ray_path_length"] >= df["distance"]).all()

    # Internal methods for setting picks and time windows
    def _set_picks_and_windows(
        self, data, mapping, param_name, label, parse_df, set_param
    ):
        """
        Internal method for setting picks and time windows from a mapping-like thing
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
            # Try to run with it
            if not hasattr(mapping, "__getitem__"):
                raise TypeError(f"{param_name} should be a mapping of {label}s")
            for key in mapping:
                if key in data.index:
                    # Index is already in the dataframe, try to overwrite the parameter
                    warnings.warn(f"Overwriting existing {label}: {key}")
                    set_param(data, key, mapping[key])
                else:
                    # Index is not in the dataframe, try to attach a new parameter
                    try:
                        self._append_data(data, key, mapping[key], set_param)
                    except IndexError:
                        raise TypeError(f"{param_name} should be a mapping of {label}s")
        return data

    def _append_data(self, data_df, index, params, set_param):
        """ internal method for appending data to StatsGroup.data """
        # make sure the event is in the catalog
        if index[1] not in self.event_station_info.index.levels[0]:
            raise KeyError(f"Event is not in the catalog: {index[1]}")

        # make sure the seed id is in the inventory
        if index[2] not in self.event_station_info.index.levels[1]:
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
        # Set the index to what we want
        df = df.reset_index().set_index(list(index))
        # Convert all of the times to timestamps
        for time in time_cols:
            df[time] = df[time].apply(to_timestamp, on_none=np.nan)
        # Get the station information
        if not set(NSLC).issubset(df.columns):
            df[list(NSLC)] = expand_seed_id(df.index.get_level_values("seed_id"))
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

    # Methods for adding data, material properties, and other coefficients
    @inplace
    def set_picks(self, picks: ChannelPickType) -> Optional["StatsGroup"]:
        """
        Method for adding picks to the ChannelInfo

        Parameters
        ----------
        picks
            Mapping containing information about picks, their phase type, and their associated metadata. The mapping
            can be a pandas DataFrame containing the following columns: [event_id, seed_id, phase, time], or a
            dictionary of the form {(phase, event_id, seed_id): obspy.UTCDateTime or obspy.Pick}.

        Other Parameters
        ----------------
        inplace
            Flag indicating whether the ChannelInfo should be modified inplace or a new copy should be returned
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
        # Loop over each of the provided phase
        for ph, tw in time_windows.items():
            if not isinstance(tw, Collection) or isinstance(tw, str):
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
            # Otherwise, set the time windows
            if (
                self.data.loc[phase_ind, "starttime"].notnull().any()
                or self.data.loc[phase_ind, "endtime"].notnull().any()
            ):
                warnings.warn(
                    "Overwriting existing time windows for one or more phases."
                )
            self.data.loc[phase_ind, "starttime"] = (
                self.data.loc[phase_ind, "time"] - tw[0]
            )
            self.data.loc[phase_ind, "endtime"] = (
                self.data.loc[phase_ind, "time"] + tw[1]
            )
        self.data = self._update_meta(self.data)
        return self

    @inplace
    def apply_defaults(self):
        """
        Method to apply default parameters to any unpopulated StatsGroup parameters

        Other Parameters
        ----------------
        inplace (bool, default=False)
            Flag indicating whether the StatsGroup should be modified inplace or a new copy should be returned
        """
        df = self.add_defaults(self.data)
        self._validate_meta_df(df)
        self.data = df
        return

    def _set_pick(self, data_df, index, pick_info, append=False):
        """ write the pick information to the dataframe """
        if isinstance(pick_info, Pick):
            # parse a pick object
            net, sta, loc, chan = get_seed_id(pick_info).split(".")
            for col in PICK_COLS:
                if col == "time":
                    data_df.loc[index, "time"] = pick_info.time.timestamp
                elif col == "pick_id":
                    data_df.loc[index, "pick_id"] = pick_info.resource_id.id
                else:
                    data_df.loc[index, col] = pick_info.__dict__[col]
            data_df.loc[index, "phase_hint"] = pick_info.__dict__["phase_hint"]
            data_df.loc[index, list(NSLC)] = [net, sta, loc, chan]
        else:
            # assign the provided pick time to the dataframe
            try:
                time = UTCDateTime(pick_info).timestamp
            except TypeError:
                raise TypeError("Pick time must be an obspy UTCDateTime")
            else:
                data_df.loc[index, "time"] = time
            # Do the nslc info
            net, sta, loc, chan = index[-1].split(".")
            data_df.loc[index, list(NSLC)] = [net, sta, loc, chan]
            if append:
                # Since there is no resource_id for the pick, create a new one
                data_df.loc[index, "pick_id"] = ResourceIdentifier().id

    # time-window-specific
    def _parse_tw_df(self, data_df, df):
        """ Add a Dataframe of time_windows to the StatsGroup """
        df = self._prep_parse_df(df, _INDEX_NAMES, ["starttime", "endtime"], data_df)
        if not (df["endtime"] > df["starttime"]).all():
            raise ValueError("time window starttime must be earlier than endtime")

        # Update the data_df
        intersect = set(df.columns).intersection(set(data_df.columns))

        def _update(row, data_df):
            if (row.name in data_df.index) and (
                not np.isnan(data_df.loc[row.name, "starttime"])
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
            net, sta, loc, chan = index[-1].split(".")
            data_df.loc[index, list(NSLC)] = [net, sta, loc, chan]
            data_df.loc[index, ["time", "pick_id"]] = [
                starttime,
                ResourceIdentifier().id,
            ]

    def _append_tw(self, data_df, index, pick_info):
        # make sure the event is in the catalog
        if index[1] not in self.event_station_info.index.levels[0]:
            raise KeyError(f"Event is not in the catalog: {index[1]}")

        # make sure the seed id is in the inventory
        if index[2] not in self.event_station_info.index.levels[1]:
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
            df.loc[:, "starttime"] = df["starttime"] - start
        if end is not None:
            df.loc[:, "endtime"] = df["endtime"] + end
        return self.new_from_dict({"data": df})

    def add_mopy_metadata(self, df):
        """
        Responsible for adding any needed metadata to df.
        """
        # add source-receiver distance
        df = self.add_source_receiver_distance(df)
        # add ray_path_lengths
        df = self.add_ray_path_length(
            df
        )  # How is this different from source-receiver distance???

        # add travel time
        df = self.add_travel_time(df)
        return df

    def add_defaults(self, df):
        """
        Populate nan values in df with default values
        """
        # add velocity
        df = self.add_velocity(df)
        # add radiation pattern corrections
        df = self.add_radiation_coeficient(df)
        # add quality factors
        df = self.add_quality_factor(df)
        # add geometric spreading factor
        df = self.add_spreading_coefficient(df)
        # add density
        df = self.add_density(df)
        # add shear modulus
        df = self.add_shear_modulus(df)
        # add free surface correction
        df = self.add_free_surface_coefficient(df)
        return df

    def add_source_receiver_distance(self, df):
        """
        Add ray path distance to each channel event pair.

        By default only a simple straight-ray distance is used. This can
        be overwritten to use a more accurate ray-path distance.
        """
        dist = self.event_station_info.copy()
        if "distance" not in df.columns:
            df["distance"] = np.nan
        dist.index.names = ("event_id", "seed_id")
        # there need to be common index names
        assert set(dist.index.names).issubset(set(df.index.names))
        # broadcast distance df to shape of df and join
        phases = list(df.index.get_level_values("phase_hint").unique().values)
        distdf = pd.concat(
            [dist for _ in range(len(phases))], keys=phases, names=["phase_hint"]
        )
        df = df.copy()
        df.update(distdf, overwrite=False)
        return df

    def add_ray_path_length(self, df, ray_length=None):
        """
        Add ray path distance to each channel event pair.

        By default only a simple straight-ray distance is used. This can
        be overwritten to use a more accurate ray-path distance.
        """
        ray_length = df["distance"] if ray_length is not None else df["distance"]
        df["ray_path_length"] = ray_length
        return df

    def add_velocity(self, df, velocity=None):
        """ Add the velocity to meta dataframe """
        if velocity is None:
            vel_map = dict(
                S=get_default_param("s_velocity"), P=get_default_param("p_velocity")
            )
            velocity = df.index.get_level_values("phase_hint").map(vel_map)
        df["velocity"] = velocity
        return df

    def add_spreading_coefficient(self, df, spreading=None):
        """ add the spreading coefficient. If None assume 1 / distance. """
        spreading = df["distance"] if spreading is None else spreading
        df["spreading_coefficient"] = spreading
        return df

    def add_radiation_coeficient(self, df, radiation_ceoficient=None):
        """ Add the factor used to correct for radiation pattern. """
        radiation_coef_map = dict(
            S=get_default_param("s_radiation_coefficient"),
            P=get_default_param("p_radiation_coefficient"),
            Noise=1.0,
        )
        rad = df.index.get_level_values("phase_hint").map(radiation_coef_map)
        df["radiation_coefficient"] = rad
        return df

    def add_quality_factor(self, df, quality_factor=None):
        if quality_factor is None:
            quality_factor = get_default_param("quality_factor")
        df["quality_factor"] = quality_factor
        return df

    def add_density(self, df, density=None):
        """
        Add density to the meta dataframe. If None, use defaults.
        """
        if density is None:
            density = get_default_param("density")
        df["density"] = density
        return df

    def add_shear_modulus(self, df, shear_modulus=None):
        """
        Add the shear modulus to the meta dataframe
        """
        if shear_modulus is None:
            shear_modulus = get_default_param("shear_modulus")
        df["shear_modulus"] = shear_modulus
        return df

    def add_free_surface_coefficient(self, df, free_surface_coefficient=None):
        """
        Add the coefficient which corrects for free surface.

        By default just uses 1/2 if the depth of the instrument is 0, else 1.
        """
        # I -know- there has to be a better way to do this, but me and pandas MultiIndexing are not getting along
        if "free_surface_coefficient" not in df.columns:
            df["free_surface_coefficient"] = np.nan
        # Get the indices of records that do not have a free_surface_coefficient specified
        subset = df.loc[df.free_surface_coefficient.isnull()].index
        if free_surface_coefficient:
            df.loc[subset, "free_surface_coefficient"] = free_surface_coefficient
        else:
            station_info = self.event_station_info
            # For my sanity's sake, make a mapping of the multi-index labels to their corresponding indices
            label_dict = {label: ind for (ind, label) in enumerate(subset.names)}
            for row in subset:
                # retrieve the depth of the station corresponding to the record
                station_depth = (
                    station_info.xs(
                        (row[label_dict["event_id"]], row[label_dict["seed_id"]]),
                        level=("event_id", "seed_id"),
                    )
                    .iloc[0]
                    .station_depth
                )
                # choose the free_surface_coefficient based on the depth
                df.loc[row, "free_surface_coefficient"] = (
                    2.0 if station_depth == 0.0 else 1.0
                )
        # if free_surface_coefficient is None:
        #     ones = pd.Series(np.zeros(len(df)), index=df.index)
        #     ones[df['station_depth'] == 0.0] = 2.0
        #     free_surface_coefficient = ones
        # df['free_surface_coefficient'] = free_surface_coefficient
        return df

    def add_travel_time(self, df):
        """
        Add the travel time of each phase to the dataframe.
        """
        return df
