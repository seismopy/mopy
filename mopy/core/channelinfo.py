"""
Class for keeping track of all metadata used by mopy.
"""
import copy
from typing import Optional, Collection, Union

import numpy as np
import obsplus
import pandas as pd
from mopy.config import get_default_param
from mopy.constants import _INDEX_NAMES
from mopy.utils import get_phase_window_df, new_from_dict
from obsplus.utils import get_distance_df, get_nslc_series
from obspy import Catalog, Inventory
from obspy.core import AttribDict


class ChannelInfo:
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
        self.distance = get_distance_df(catalog, inventory)
        df = self._get_meta_df(catalog, inventory, phases=phases)
        self.data = df
        # add sampling rate to stats
        sampling_rate = self.data["sampling_rate"].unique()[0]
        self._stats = AttribDict(sampling_rate=sampling_rate)

    def _get_meta_df(self, catalog, inventory, phases=None):
        """
        Return a dataframe containing pick/duration info.

        Uses defaults, all of which can be overwritten in the config file.

        Columns contain info on time window start/stop as well as traces.
        Index is multilevel using (event_id, phase, seed_id).
        """
        # calculate source-receiver distance.
        dist_df = self.distance
        sta_df = obsplus.stations_to_df(inventory)
        # create a list of sampling rates per channel
        sample_rate = sta_df.set_index(get_nslc_series(sta_df))["sample_rate"]
        df_list = []  # list for gathering dataframes
        kwargs = dict(dist_df=dist_df, phases=phases, sample_rate=sample_rate)
        for event in catalog:
            df_list.extend(list(self._get_event_meta(event, **kwargs)))
        # concat df and perform sanity checks
        df = pd.concat(df_list, sort=True)
        # before adding metadata, there should be no NaNs
        assert not df.isnull().any().any()
        # add station_depth, station_azimuth, and station_dip
        df = self._join_station_info(df, obsplus.stations_to_df(inventory))
        # then get duration/pick dataframes for each event
        # add metadata and validate
        df = self.add_metadata(df.sort_index())
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
        assert (df["tw_end"] >= df["tw_start"]).all(), "negative lengths found!"
        # if any windows have the same start as end time NaN start/end times
        is_zero_len = df["tw_end"] == df["tw_start"]
        df.loc[is_zero_len, ["tw_end", "tw_start"]] = np.NaN
        # filter df to only include phases for which data exist, warn
        df_noise = self._get_noise_windows(df, noise_df)
        df = df.set_index(list(_INDEX_NAMES))
        return df_noise, df

    def _join_station_info(self, df, station_df):
        """ Joins some important station info to dataframe. """
        col_map = dict(
            depth="station_depth", azimuth="station_azimuth", dip="station_dip"
        )
        sta_df = station_df.set_index("seed_id")[list(col_map)]
        return df.join(sta_df.rename(columns=col_map), how="left")

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
        df = get_phase_window_df(
            event, min_duration=min_duration, channel_codes=set(min_duration.index)
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
            # get parameters for determining noise windows start and stop
            endtime = get_default_param("noise_end_before_p", obj=self)
            min_noise_dur = get_default_param("noise_min_duration", obj=self)
            largest_window = (phase_df["tw_end"] - phase_df["tw_start"]).max()
            min_duration = max(min_noise_dur, largest_window)
            # set start and stop for noise window
            noise_df["tw_end"] = phase_df["tw_start"].min() - endtime
            noise_df["tw_start"] = noise_df["tw_end"] - min_duration
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
        assert (df["tw_end"] > df["tw_start"]).all()
        # ray_paths should all be at least as long as the source-receiver dist
        assert (df["ray_path_length"] >= df["distance"]).all()

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
            df.loc[:, "tw_start"] = df["tw_start"] - start
        if end is not None:
            df.loc[:, "tw_end"] = df["tw_end"] + end
        return self.new_from_dict({"data": df})

    # customizable methods

    def add_metadata(self, df):
        """
        Responsible for adding any needed metadata to df.
        """
        # add source-receiver distance
        df = self.add_source_receiver_distance(df)
        # add ray_path_lengths
        df = self.add_ray_path_length(df)
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
        dist = self.distance.copy()
        dist.index.names = ("event_id", "seed_id")
        # there need to be common index names
        assert set(dist.index.names).issubset(set(df.index.names))
        # broadcast distance df to shape of df and join
        phases = list(df.index.get_level_values("phase_hint").unique().values)
        distdf = pd.concat(
            [dist for _ in range(len(phases))], keys=phases, names=["phase_hint"]
        )
        return df.join(distdf)

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
        if free_surface_coefficient is None:
            ones = pd.Series(np.zeros(len(df)), index=df.index)
            ones[df["station_depth"] == 0.0] = 2.0
            free_surface_coefficient = ones
        df["free_surface_coefficient"] = free_surface_coefficient
        return df

    def add_travel_time(self, df):
        """
        Add the travel time of each phase to the dataframe.
        """
        return df

    new_from_dict = new_from_dict

    def copy(self):
        """ Create a copy of ChannelInfo, dont copy nested traces. """
        # first create a shallow copy, then deep copy when needed
        # if traces are here make sure they aren't copied
        df = self.data.copy()
        distance = self.distance.copy()
        # now attach copied stuff
        new = copy.copy(self)
        new.data = df
        new.distance = distance
        new._stats = copy.deepcopy(new._stats)
        return new

    # --- dunders

    def __str__(self):
        return str(self.data)

    __repr__ = __str__
