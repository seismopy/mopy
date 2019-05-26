"""
Class for keeping track of all metadata used by mopy.
"""
import copy
import warnings
from typing import Dict, Union, Optional, Collection, Tuple
from functools import singledispatch

import numpy as np
import obsplus
import pandas as pd
from obsplus.utils import get_distance_df
from obsplus.events.utils import get_seed_id
from obspy import Catalog, Inventory, Stream
from obspy.core import AttribDict, UTCDateTime
from obspy.core.event import Pick, ResourceIdentifier

from mopy.config import get_default_param
from mopy.constants import _INDEX_NAMES, CHAN_COLS, CHAN_DTYPES, NSLC, PICK_COLS
from mopy.exceptions import DataQualityError, NoPhaseInformationError
from mopy.utils import get_phase_window_df, expand_seed_id


CHAN_PICKS = Union[str, pd.DataFrame, Dict[Tuple[str, str, str], Union[UTCDateTime, Pick]]]


class ChannelInfo:
    """
    Class for creating information about each channel.

    Parameters
    ----------
    catalog
        Data containing information about the events.
    inventory
        Station data.
    st_dict
        A dictionary of the form {str(event_id): stream}.
        #TODO expand this to include generic waveform clients
    motion_type
        A string indicating the ground
    phases
        if Not None, only include phases provided
    """

    def __init__(
            self,
            catalog: Catalog,
            inventory: Inventory,
            st_dict: Dict[str, Stream],
            motion_type: Union[str, Dict[str, str]] = 'velocity',
            phases: Optional[Collection[str]] = None,
    ):
        # check inputs
        st_dict, catalog = self._validate_inputs(catalog, st_dict)
        # get a df of all input data, perform sanity checks
        self.event_station_info = get_distance_df(catalog, inventory)
        self.event_station_info.index.names = ["event_id", "seed_id"]
        self._join_station_info(obsplus.stations_to_df(inventory))
        df = self._get_meta_df(catalog, st_dict, phases=phases)
        self.data = df
        # add sampling rate to stats
        self._stats = AttribDict(motion_type=motion_type)
        if len(self):
            sampling_rate = self.data['sampling_rate'].unique()[0]  # This is an interesting dilemma... would this have to be called later? And where is it actually getting the sampling rate? Should this be deleted now that the streams are not going to be added to the ChannelInfo
            self._stats["sampling_rate"] = sampling_rate

    # Methods for adding data, material properties, and other coefficients
    def set_picks(self, picks: CHAN_PICKS, inplace: bool = False) -> Optional['ChannelInfo']:
        """
        Method for adding picks to the ChannelInfo

        Parameters
        ----------
        picks
            Mapping containing information about picks, their phase type, and their associated metadata. The mapping
            can be a pandas DataFrame containing the following columns: [event_id, seed_id, phase, time], or a
            dictionary of the form {(phase, event_id, seed_id): obspy.UTCDateTime or obspy.Pick}.
        inplace
            Flag indicating whether the ChannelInfo should be modified inplace or a new copy should be returned
        """
        if not inplace:
            ci = self.copy()
        else:
            ci = self
        data_df = ci.data
        if isinstance(picks, str):
            try:
                picks = pd.read_csv(picks)
            except TypeError:
                raise TypeError("picks should be a mapping of pick times")
            self._parse_pick_df(data_df, picks)
        elif isinstance(picks, pd.DataFrame):
            self._parse_pick_df(data_df, picks)
        else:
            # Try to run with it
            if not hasattr(picks, "__getitem__"):
                raise TypeError("picks should be a mapping of pick times")
            for key in picks:
                if key in self.data.index:
                    # Index is already in the dataframe, try to overwrite the pick
                    warnings.warn(f"Overwriting existing pick: {key}")
                    self._set_pick(data_df, key, picks[key])
                else:
                    # Index is not in the dataframe, try to attach a new pick
                    try:
                        self._append_pick(data_df, key, picks[key])
                    except IndexError:
                        raise TypeError('picks should be a mapping of pick times')
        ci.data = self._update_meta(ci.data)
        return ci

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
            msg = f'missing data for event ids: {original_ids - filtered_ids}'
            warnings.warn(msg)
        return st_dict, cat

    def _get_meta_df(self, catalog, st_dict, phases=None):
        """
        Return a dataframe containing pick/duration info.

        Columns contain info on time window start/stop as well as traces.
        Index is multilevel using (event_id, phase, seed_id).
        """
        # calculate source-receiver distance.
        dist_df = self.event_station_info
        # get a set of all available channel codes
        channel_codes = {tr.id for eid, st in st_dict.items() for tr in st}
        sr_dict = {
            tr.stats.sampling_rate for st in st_dict.values() for tr in st
        }

        # for now assert all sampling rates are uniform; consider handling
        # multiple sampling rates in the future (it will be tricky!)
        assert len(sr_dict) == 1, "uniform sampling rates required"
        sampling_rate = list(sr_dict)[0]

        df_list = []  # list for gathering dataframes
        kwargs = dict(st_dict=st_dict, dist_df=dist_df, channel_codes=channel_codes,
                      phases=phases, sampling_rate=sampling_rate)
        for event in catalog:
            try:
                signal_df, noise_df = self._get_event_meta(event, **kwargs)
            except NoPhaseInformationError:
                df = pd.DataFrame(columns=CHAN_COLS + _INDEX_NAMES)  # , dtype=CHAN_DTYPES)
                return df.set_index(list(_INDEX_NAMES))
            except Exception:
                warnings.warn(f'failed on {event}')
            else:
                df_list.extend([signal_df, noise_df])
        # concat df and perform sanity checks
        df = pd.concat(df_list, sort=True)
        # before adding metadata, there should be no NaNs
        assert not df.isnull().any().any()
        df = self._update_meta(df)
        return df

    def _update_meta(self, df: pd.DataFrame) -> pd.DataFrame:
        # get duration/pick dataframes for each event
        # add metadata and validate
        df = self.add_metadata(df.sort_index())
        self._validate_meta_df(df)
        return df

    def _get_event_meta(self, event, st_dict, dist_df, channel_codes, sampling_rate,
                        phases):
        """ For an event create dataframes of signal windows and noise windows. """
        event_id = str(event.resource_id)
        st = st_dict[event_id]
        trace_dict = {tr.id: tr for tr in st}
        sr_dict = {key: tr.stats.sampling_rate
                   for key, tr in trace_dict.items()}
        # make sure there is one trace per seed_id
        assert len(trace_dict) == len(st), "duplicate traces found!"
        # get time windows
        df = self._get_event_phase_window(
            event, dist_df, channel_codes=channel_codes, sampling_rate=sampling_rate
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
        df["trace"] = df["seed_id"].map(trace_dict)
        # add sample rates of stream
        df['sampling_rate'] = df['seed_id'].map(sr_dict)
        # filter df to only include phases for which data exist, warn
        zero_duration = (df["tw_end"] - df["tw_start"]) == 0.0
        no_trace = df["trace"].isnull()
        should_drop = zero_duration | no_trace
        no_data = df[df["trace"].isnull() & zero_duration]
        if should_drop.any():
            chans = no_data['seed_id']
            msg = f"no data for event: {event_id} on channels:\n{chans}"
            warnings.warn(msg)
            df = df[~should_drop]
        df_noise = self._get_noise_windows(df, noise_df)
        df = df.set_index(list(_INDEX_NAMES))
        return df_noise, df

    def _join_station_info(self, station_df: pd.DataFrame):
        """ Joins some important station info to the event_station_info dataframe. """
        col_map = dict(depth='station_depth', azimuth='station_azimuth',
                       dip='station_dip')
        sta_df = station_df.set_index('seed_id')[list(col_map)]
        # There has to be a more efficient pandas way to do this, but I'm just not having any luck at the moment
        for num, row in sta_df.iterrows():
            self.event_station_info.loc[(slice(None), num), "station_depth"] = row.depth
            self.event_station_info.loc[(slice(None), num), "station_azimuth"] = row.azimuth
            self.event_station_info.loc[(slice(None), num), "station_dip"] = row.dip
        return

    def _get_event_phase_window(self, event, dist_df, channel_codes, sampling_rate):
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
        min_duration = np.maximum(min_dur_dist, min_dur_samps)
        _percent_taper = get_default_param("percent_taper", obj=self)
        # get dataframe
        if not len(event.picks):
            raise NoPhaseInformationError()
        df = get_phase_window_df(
            event,
            min_duration=min_duration,
            channel_codes=channel_codes,
            buffer_ratio=_percent_taper,
        )
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
            starttimes = [tr.stats.starttime.timestamp for tr in noise_df["trace"]]
            max_duration = (phase_df.tw_end - phase_df.tw_start).max()
            noise_df["tw_start"] = starttimes
            noise_df["tw_end"] = noise_df["tw_start"] + max_duration
        else:
            # else use either the noise window defined for a specific station
            # or, if a station has None, use the noise window with the earliest
            # start time
            ser_min = df.loc[df.tw_start.idxmin()]
            t1, t2 = ser_min.tw_start, ser_min.tw_end
            # drop columns on df and noise df to facilitate merge
            df = df[["network", "station", "tw_start", "tw_end"]]
            noise_df = noise_df.drop(columns=["tw_start", "tw_end"])
            noise_df = noise_df.merge(df, how="left", on=["network", "station"])
            # fill nan
            noise_df = noise_df.fillna({"tw_start": t1, "tw_end": t2})
        # set time between min and max
        noise_df["time"] = noise_df[["tw_start", "tw_end"]].mean(axis=1)
        # make sure there are no null values
        out = noise_df.set_index(list(_INDEX_NAMES))
        # drop any duplicate indices
        return out.loc[~out.index.duplicated()]

    def _validate_meta_df(self, df):
        """ Perform simple checks on meta df that should always hold True. """
        # there should be no duplicated indices
        assert not df.index.duplicated().any()
        # all start times should be less than their corresponding end-time
        if df["tw_end"].notnull().any():
            assert (df["tw_end"] > df["tw_start"]).all()
        # ray_paths should all be at least as long as the source-receiver dist
        assert (df["ray_path_length"] >= df["distance"]).all()

    def _parse_pick_df(self, data_df, df):  # This could probably be cleaned up a little bit?
        """ Add a Dataframe of picks to the ChannelInfo """
        # If the provided data frame is multi-indexed, just clear it
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        # Set the index to what we want
        df = df.set_index(list(_INDEX_NAMES))
        # Convert all of the times to obspy objects (do we actually want timestamps? probably?)
        df["time"] = df.time.apply(UTCDateTime)
        # Get the station information
        if not set(NSLC).issubset(df.columns):
            df[list(NSLC)] = expand_seed_id(df.index.get_level_values("seed_id"))
        # Get the resource_id
        if ("resource_id" in df.columns) and ("pick_id" in df.columns):
            if not df.resource_id == df.pick_id:
                raise KeyError("resource_id and pick_id must be identical for a pick DataFrame")
        elif "resource_id" in df.columns:
            df.rename(columns={"resource_id": "pick_id"})
        else:
            # resource_ids were not provided, try to make one (but make sure not to overwrite an existing one!)
            def _make_resource_id(row, data_df):
                if row.name in data_df.index:
                    return data_df.loc[row.name].pick_id
                else:
                    return ResourceIdentifier().id
            df["pick_id"] = df.apply(_make_resource_id, axis=1, data_df=data_df)
        # Had to get creative to overwrite the existing dataframe, there may be a cleaner way to do this
        intersect = set(df.columns).intersection(set(data_df.columns))
        diff = set(df.columns).difference(set(data_df.columns))
        # For columns that already exist, update their values
        for ind, row in df.iterrows():
            if ind in data_df.index:
                warnings.warn(f"Overwriting existing pick: {ind}")
            data_df.loc[ind, intersect] = row[intersect]
        # Create new columns for the ones that don't exist
        for col in diff:
            data_df.loc[df.index, col] = df[col]

    def _set_pick(self, data_df, index, pick_info, append=False):
        """ write the pick information to the dataframe """
        if isinstance(pick_info, Pick):
            # parse a pick object
            net, sta, loc, chan = get_seed_id(pick_info).split(".")
            for col in PICK_COLS:
                if col == "pick_id":
                    data_df.loc[index, "pick_id"] = pick_info.resource_id.id
                else:
                    data_df.loc[index, col] = pick_info.__dict__[col]
            data_df.loc[index, "phase_hint"] = pick_info.__dict__["phase_hint"]
            data_df.loc[index, list(NSLC)] = [net, sta, loc, chan]
        else:
            # assign the provided pick time to the dataframe
            try:
                time = UTCDateTime(pick_info)
            except TypeError:
                raise TypeError("Pick time must be an obspy UTCDateTime")
            else:
                data_df.loc[index, "time"] = time
            # Do the nslc info
            net, sta, loc, chan = index[-1].split(".")
            data_df.loc[index, list(NSLC)] = [net, sta, loc, chan]
            data_df.loc[index, "phase_hint"] = index[0]
            if append:
                # Since there is no resource_id for the pick, create a new one
                data_df.loc[index, "pick_id"] = ResourceIdentifier().id

    def _append_pick(self, data_df, index, pick_info):
        # make sure the event is in the catalog
        if index[1] not in self.event_station_info.index.levels[0]:
            raise KeyError(f"Event is not in the catalog: {index[1]}")

        # make sure the seed id is in the inventory
        if index[2] not in self.event_station_info.index.levels[1]:
            raise KeyError(f"seed_id is not in the inventory: {index[2]}")

        # extract the necessary information from the pick and append it to the data_df
        self._set_pick(data_df, index, pick_info, append=True)

    # Customizable methods
    def add_metadata(self, df):
        """
        Responsible for adding any needed metadata to df.
        """
        # add source-receiver distance
        df = self.add_source_receiver_distance(df)
        # add ray_path_lengths
        df = self.add_ray_path_length(df) # How is this different from source-receiver distance???
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
        # add travel time
        df = self.add_travel_time(df)
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
                station_depth = station_info.xs(
                    (row[label_dict["event_id"]], row[label_dict["seed_id"]]),
                    level=("event_id", "seed_id")).iloc[0].station_depth
                # choose the free_surface_coefficient based on the depth
                df.loc[row, "free_surface_coefficient"] = 2.0 if station_depth == 0.0 else 1.0
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

    def copy(self):
        """ Create a copy of ChannelInfo, dont copy nested traces. """
        # first create a shallow copy, then deep copy when needed
        # if traces are here make sure they aren't copied
        if 'trace' in self.data.columns:
            df = self.data.drop(columns='trace').copy()
            df['trace'] = self.data['trace']
        else:
            df = self.data.copy()
        distance = self.event_station_info.copy()
        # now attach copied stuff
        new = copy.copy(self)
        new.data = df
        new.event_station_info = distance
        new._stats = copy.deepcopy(new._stats)
        return new

    # --- dunders

    def __str__(self):
        return str(self.data)

    def __len__(self):
        return len(self.data)

    __repr__ = __str__
