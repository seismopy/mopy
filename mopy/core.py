"""
Core classes
"""
import copy
import pickle
import warnings
from pathlib import Path
from typing import Optional, Union, Dict, Collection, TypeVar, Type, Tuple

import numpy as np
import obspy
import pandas as pd
import obsplus
from obsplus.utils import get_distance_df
from obsplus.constants import NSLC
from obspy import Stream, Catalog, Inventory
from obspy.core.util import AttribDict
from scipy.fftpack import next_fast_len

from mopy.config import get_default_param
from mopy.constants import MOTION_TYPES
from mopy.smooth import konno_ohmachi_smoothing as ko_smooth
from mopy.sourcemodels import fit_model
from mopy.utils import get_phase_window_df, _source_process
from mopy.exceptions import DataQualityError

# A map to get a function which returns the constant to multiply to perform
# temporal integration/division in the freq. domain
motion_maps = {
    ("velocity", "displacement"): lambda freqs: 1 / (2 * np.pi * freqs),
    ("velocity", "acceleration"): lambda freqs: 2 * np.pi * freqs,
    ("acceleration", "velocity"): lambda freqs: 1 / (2 * np.pi * freqs),
    ("acceleration", "displacement"): lambda freqs: 1 / ((2 * np.pi * freqs) ** 2),
    ("displacement", "velocity"): lambda freqs: 2 * np.pi * freqs,
    ("displacement", "acceleration"): lambda freqs: (2 * np.pi * freqs) ** 2,
}

_INDEX_NAMES = ("phase_hint", "event_id", "seed_id")

# This is type annotation to specify subclass outputs of parent methods on
# the DataFrameGroupBase type
DFG = TypeVar('DF_TYPE', bound='DataFrameGroupBase')


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
        st_dict, catalog = self._validate_inputs(catalog, inventory, st_dict)
        # get a df of all input data, perform sanity checks
        self.distance = get_distance_df(catalog, inventory)
        df = self._get_meta_df(catalog, inventory, st_dict, phases=phases)
        self.data = df
        # add sampling rate to stats
        sampling_rate = self.data['sampling_rate'].unique()[0]
        self._stats = AttribDict(motion_type=motion_type,
                                 sampling_rate=sampling_rate)

    def _validate_inputs(self, events, stations, waveforms
                         ) -> Tuple[Dict[str, Stream], Catalog]:
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

    def _get_meta_df(self, catalog, inventory, st_dict, phases=None):
        """
        Return a dataframe containing pick/duration info.

        Columns contain info on time window start/stop as well as traces.
        Index is multilevel using (event_id, phase, seed_id).
        """
        # calculate source-receiver distance.
        dist_df = self.distance
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
            except Exception:
                warnings.warn(f'failed on {event}')
            else:
                df_list.extend([signal_df, noise_df])
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

    def _join_station_info(self, df, station_df):
        """ Joins some important station info to dataframe. """
        col_map = dict(depth='station_depth', azimuth='station_azimuth',
                       dip='station_dip')
        sta_df = station_df.set_index('seed_id')[list(col_map)]
        return df.join(sta_df.rename(columns=col_map), how='left')

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
        # drop any duplicate indicies
        return out.loc[~out.index.duplicated()]

    def _validate_meta_df(self, df):
        """ Perform simple checks on meta df that should always hold True. """
        # there should be no duplicated indices
        assert not df.index.duplicated().any()
        # all start times should be less than their corresponding end-time
        assert (df["tw_end"] > df["tw_start"]).all()
        # ray_paths should all be at least as long as the source-receiver dist
        assert (df["ray_path_length"] >= df["distance"]).all()

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
            ones[df['station_depth'] == 0.0] = 2.
            free_surface_coefficient = ones
        df['free_surface_coefficient'] = free_surface_coefficient
        return df

    def add_travel_time(self, df):
        """
        Add the travel time of each phase to the dataframe.
        """
        return df

    def copy(self):
        """ Create a copy of ChannelInfo, dont copy nested traces. """
        # first creat a shallow copy, then deep copy when needed
        # if traces are here make sure they arent copied
        if 'trace' in self.data.columns:
            df = self.data.drop(columns='trace').copy()
            df['trace'] = self.data['trace']
        else:
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


class DataFrameGroupBase:
    """ Base class for TraceGroup and SpectrumGroup. """
    channel_info: ChannelInfo
    stats: AttribDict
    data: pd.DataFrame

    @property
    def meta(self):
        return self.channel_info.data

    @meta.setter
    def meta(self, item):
        self.channel_info.data = item

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

    def new_from_dict(self: DFG, update: dict) -> DFG:
        """
        Create a new object from a dict input to the old object.
        """
        copy = self.copy()
        copy.__dict__.update(update)
        return copy

    def in_prococessing(self, name):
        """
        Return True in name is a substring of any of the processing strings.
        """
        proc = getattr(self.stats, "processing", ())
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
        return self.new_from_dict({'data': df})

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
        seed_id = old_index.get_level_values('seed_id').to_series()
        nslc = seed_id.str.split('.', expand=True)
        nslc.columns = list(NSLC)
        # add old index values to keep back
        nslc['phase_hint'] = old_index.get_level_values('phase_hint').values
        nslc['event_id'] = old_index.get_level_values('event_id').values
        cols = ['phase_hint', 'event_id', 'network', 'station', 'location',
                'channel']
        return pd.MultiIndex.from_arrays(nslc[cols].values.T, names=cols)

    def _get_collapsed_index(self) -> pd.Index:
        """ collapse and index that has """
        pass


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
        # TODO this requires more thought
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


class TraceGroup(DataFrameGroupBase):
    """
    Class for storing time series as pandas dataframes.

    Will also copy and update channel info.
    """

    def __init__(self, channel_info: ChannelInfo):
        """ init instance. """
        super().__init__()
        self.channel_info = channel_info.copy()
        self.stats = channel_info._stats
        self.data = self._make_trace_df(self.channel_info.data)
        # drop traces to avoid unwanted copying of many large(ish) objects
        self.channel_info.data = self.channel_info.data.drop(columns='trace')

    def _make_trace_df(self, phase_df):
        """
        Make the data arrays.
        """
        # get time rep. in dataframe
        sampling_rate = phase_df['sampling_rate'].unique()[0]
        num_samples = (phase_df['tw_end'] - phase_df['tw_start']) * sampling_rate
        array_len = next_fast_len(int(num_samples.max() + 1))
        time = np.arange(0, array_len) * (1. / sampling_rate)
        # apply preprocessing to each trace
        traces = phase_df.apply(self.process_trace, axis=1)
        # create numpy array, fill with data
        values = np.zeros((len(phase_df.index), len(time)))
        for i, trace in enumerate(traces.values):
            values[i, 0:len(trace.data)] = trace.data
        # init df from filled values
        df = pd.DataFrame(values, index=phase_df.index, columns=time)
        # set name of columns
        df.columns.name = 'time'
        # add original lengths to the channel_info
        lens = traces.apply(lambda x: len(x.data))
        self.channel_info.data['sample_count'] = lens
        # apply time-domain pre-processing
        df = self.process_trace_dataframe_hook(df)
        return df

    def process_trace(self, ser):
        """
        Process trace.

        By default will simply detrend and taper window length in prep for
        fft.

        Notes
        -----
        Many phase windows may share the same trace; be sure to copy the data
        before modification.

        The parameter ser is a row of the channel info dataframe.
        """
        # slice out time of interest
        t1, t2 = obspy.UTCDateTime(ser.tw_start), obspy.UTCDateTime(ser.tw_end)
        tr = ser.trace.slice(starttime=t1, endtime=t2).copy()
        tr.detrend("linear")
        # apply taper for ffts
        taper_ratio = get_default_param('PERCENT_TAPER', obj=self.channel_info)
        tr.taper(max_percentage=taper_ratio)
        return tr.data

    def process_trace_dataframe_hook(self, df):
        """
        A hook for processing a trace dataframe.

        If not overwritten will simply return the dataframe.
        """
        return df


class SpectrumGroup(DataFrameGroupBase):
    """
    A class to encompass many catalog sources.
    """

    min_samples = 60  # required number of samples per phase
    _default_velocity = {"P": 4000, "S": 2400}
    _default_radiation = {"P": 0.44, "S": 0.6}
    _default_quality_factor = 250
    _pre_taper = 0.10  # percentage of window added before/after phase to taper
    _max_taper_percentage = 0.05
    # The number of seconds per source-receiver distance to require for
    # phase windows, while still requiring a min number of samples.
    _seconds_per_m = 0.000_03
    # the time domain data, useful for debugging
    _td_data = None
    # DF for storing info about source parameters
    source_df = None

    def __init__(self, trace_group: TraceGroup):
        # check inputs
        self.channel_info = trace_group.channel_info.copy()
        self.stats = trace_group.stats.copy()
        # set stats
        self.data = self._make_spectra_df(trace_group)
        # init empty source dataframe
        self.source_df = pd.DataFrame(index=self.meta.index)

    def _make_spectra_df(self, trace_group):
        """
        Make the data arrays.
        """
        df_freq = self.to_freq_domain(trace_group.data)
        return self.process_spectra_dataframe_hook(df_freq)

    def to_freq_domain(self, df):
        """
        Convert the dataframe from TraceGroup (in time domain) to freq. domain.

        Parameters
        ----------
        df

        Notes
        -----
        The fft needs to be scaled by the sampling rate in order to emulate a
        continuous transform.
        """
        # get fft frequencies
        sampling_rate = self.meta['sampling_rate'].values[0]
        freqs = np.fft.rfftfreq(df.values.shape[-1], 1 / sampling_rate)
        # perform fft, divide by sampling rate and double non 0 freq. to account
        # for only using the positive frequencies

        fft = np.fft.rfft(df.values, axis=1)

        # fft[1:] *= 2
        # divide by sampling rate so it behaves like continuous transform
        # fft /= sampling_rate

        df = pd.DataFrame(fft, index=df.index, columns=freqs)
        # set name of column
        df.columns.name = 'frequency'
        df = df.divide(self.meta['sample_count'], axis=0)
        return df

    # --- SpectrumGroup hooks

    def post_source_function_hook(self):
        """ A hook that gets called after each source function. """
        pass

    def process_spectra_dataframe_hook(self, df):
        """
        Process the frequency domain dataframe.

        Generally this includes adjusting for geometric spreading,

        masking data under noise levels
        """
        return df

    @_source_process
    def ko_smooth(self, frequencies: Optional[np.ndarray] = None) -> 'SpectrumGroup':
        """
        Return new SourceGroup which has konno-ohmachi smoothing applied to it.

        Parameters
        ----------
        frequencies
            Frequencies to use to re-sample the array.

        Returns
        -------

        """
        # TODO add other types of smoothing
        # get inputs for smoothing
        vals = self.data.values
        freqs = self.data.columns.values
        smoothed = ko_smooth(
            vals, frequencies=freqs, center_frequencies=frequencies, normalize=True
        )
        freqs_out = frequencies if frequencies is not None else freqs
        df = pd.DataFrame(smoothed, index=self.data.index, columns=freqs_out)
        return self.new_from_dict({"data": df})

    @_source_process
    def subtract_phase(
            self, phase_hint: str = "Noise", drop: bool = True, negative_nan=True
    ) -> 'SpectrumGroup':
        """
        Return new SourceGroup with one phase subtracted from the others.

        Parameters
        ----------
        phase_hint
            The phase to subtract. By default use noise.
        drop
            If True drop the subtracted phase, otherwise all its rows will be
            0s.
        negative_nan
            If True set all values below 0 to NaN.
        """
        # get inputs for smoothing
        assert phase_hint in self.data.index.get_level_values("phase_hint")
        subtractor = self.data.loc[phase_hint]
        out = []
        names = []
        for phase_name, df in self.data.groupby(level="phase_hint"):
            # TODO is there such thing as a left subtract?
            if phase_name == phase_hint and drop:
                continue
            ddf = df.loc[phase_name]
            out.append(ddf - subtractor.loc[ddf.index])
            names.append(phase_name)
        df = pd.concat(out, keys=names, names=["phase_hint"])
        if negative_nan:
            df[df < 0] = np.NaN
        return self.new_from_dict({"data": df})

    @_source_process
    def mask_by_phase(self, phase_hint: str = "Noise", multiplier=1, drop=True) -> 'SpectrumGroup':
        """
        Return new SourceGroup masked against another.

        This essentially compares a phase with all other pertinent data and
        masks all values where the first phase is less than the second with NaN.

        By default this will set all values in the signal phases to NaN if they
        are below the noise.

        Parameters
        ----------
        phase_hint
            The phase to subtract. By default use noise.
        multiplier
            A value to multiply the mask by. For example, this can be used
            to mask all values less than 2x the noise.
        drop
            If True drop the subtracted phase, otherwise all its rows will be
            0s.
        """
        # get inputs for smoothing
        assert phase_hint in self.data.index.get_level_values("phase_hint")
        masker = self.data.loc[phase_hint] * multiplier
        out = []
        names = []
        for phase_name, df in self.data.groupby(level="phase_hint"):
            if phase_name == phase_hint and drop:
                continue
            ddf = df.loc[phase_name]
            out.append(ddf.mask(ddf <= masker.loc[ddf.index]))
            names.append(phase_name)
        sdict = {"data": pd.concat(out, keys=names, names=["phase_hint"])}
        return self.new_from_dict(sdict)

    @_source_process
    def normalize(self, by="station"):
        """
        Normalize phases of df as if they contained the same number of samples.

        This normalization is necessary because the spectra are taken from
        time windows of different lengths. Without normalization
        this results in longer phases being potentially larger than if they all
        contained the same number of samples before zero-padding.

        Parameters
        ----------
        by
            Some useful description.
        """
        # TODO check if this is correct (may be slightly off)
        df = self.data
        assert df.index.names == _INDEX_NAMES
        # get proper normalization factor for each row
        meta = self.meta.loc[df.index]
        group_col = meta[by]
        tw1, tw2 = meta["tw_start"], meta["tw_end"]
        samps = ((tw2 - tw1) * self.stats.sampling_rate).astype(int)
        min_samps = group_col.map(samps.groupby(group_col).min())
        norm_factor = (min_samps / samps) ** (1 / 2.0)
        # apply normalization factor
        normed = self.data.mul(norm_factor, axis=0)
        return self.new_from_dict(dict(data=normed))

    @_source_process
    def correct_attenuation(self, quality_facotor=None, drop=True):
        """
        Correct the spectra for intrinsic attenuation.

        Parameters
        ----------
        drop
            If True drop all NaN rows (eg Noise phases)
        """
        df, meta = self.data, self.meta
        required_columns = {"velocity", "quality_factor", "distance"}
        assert set(meta.columns).issuperset(required_columns)
        if quality_facotor is None:
            quality_facotor = meta["quality_factor"]
        # get vectorized q factors
        num = np.pi * meta["distance"]
        denom = quality_facotor * meta["velocity"]
        f = df.columns.values
        factors = np.exp(-np.outer(num / denom, f))
        # apply factors to data
        out = df / factors
        # drop NaN if needed
        if drop:
            out = out[~out.isnull().all(axis=1)]
        return self.new_from_dict(dict(data=out))

    @_source_process
    def correct_radiation_pattern(self, radiation_pattern=None, drop=True):
        """
        Correct for radiation pattern.

        Parameters
        ----------
        radiation_pattern
            A radiation pattern coefficient or broadcastable to data. If None
            uses the default.
        drop
            If True drop any rows without a radiation_pattern.

        Notes
        -----
        By default the radiation pattern coefficient for noise is 1, so the
        noise phases will propagate unaffected.
        """
        if radiation_pattern is None:
            radiation_pattern = self.meta["radiation_coefficient"]
        df = self.data.divide(radiation_pattern, axis=0)
        if drop:
            df = df[~df.isnull().all(axis=1)]
        return self.new_from_dict({"data": df})

    @_source_process
    def correct_free_surface(self, free_surface_coefficient=None):
        """
        Correct for stations being on a free surface.

        If no factor is provided uses the one in channel_info.

        Parameters
        ----------
        free_surface_coefficient
        """
        if free_surface_coefficient is None:
            free_surface_coefficient = self.meta['free_surface_coefficient']
        df = self.data.multiply(free_surface_coefficient, axis=0)
        return self.new_from_dict({'data': df})

    @_source_process
    def correct_spreading(self, spreading_coefficient=None):
        """
        Correct for geometric spreading.
        """

        if spreading_coefficient is None:
            spreading_coefficient = self.meta["spreading_coefficient"]

        df = self.data.multiply(spreading_coefficient, axis=0)
        return self.new_from_dict({"data": df})

    @_source_process
    def to_motion_type(self, motion_type: str):
        """
        Convert from one ground motion type to another.

        Simply uses spectral division/multiplication and zeros the zero
        frequency. Returns a copy.

        Parameters
        ----------
        motion_type
            Either "displacement", "velocity", or "acceleration".
        """
        # make sure motion type is supported
        if motion_type.lower() not in MOTION_TYPES:
            msg = f"{motion_type} is not in {MOTION_TYPES}"
            raise ValueError(msg)

        current_motion_type = self.stats.motion_type
        if current_motion_type == motion_type:
            return self.copy()

        factor = motion_maps[(current_motion_type, motion_type)]
        df = self.data * factor(self.data.columns)
        # zero freq. 0 in order to avoid inf
        df[0] = 0
        # make new stats object and return
        stats = self.stats.copy()
        stats.motion_type = motion_type
        return self.new_from_dict({"data": df, "stats": stats})

    # --- functions model fitting

    @_source_process
    def fit_source_model(self, model="brune", fit_noise=False, **kwargs):
        """
        Fit the spectra to a selected source model.

        For more fine-grained control over the inversion see the sourcemodels
        module.

        Parameters
        ----------
        model
            The model, or sequence of models, to fit to the data
        fit_noise
            If True, also fit to the noise, else drop before fitting.

        Returns
        -------

        """
        fit = fit_model(self, model=model, fit_noise=fit_noise, **kwargs)
        return self.new_from_dict(dict(fit_df=fit))

    @_source_process
    def calc_simple_source(self):
        """
        Calculate source parameters from the spectra directly.

        Rather than fitting a model this simply assumes fc is the frequency
        at which the max of the velocity spectra occur and omega0 is the mean
        of all values less than fc.

        The square of the velocity spectra is also added to the source_df.

        These are added to the source df with the group: "simple"
        """
        sg = self.abs()
        # warn/ if any of the pre-processing steps have not occurred
        self._warn_on_missing_process()  # TODO maybe this should raise?
        # get displacement and velocity spectra
        disp = sg.to_motion_type("displacement")
        vel = sg.to_motion_type("velocity")
        # estimate fc as max of velocity (works better when smoothed of course)
        fc = vel.data.idxmax(axis=1)
        lt_fc = np.greater.outer(fc, disp.data.columns.values)
        # create mask of NaN for any values greater than fc
        mask = lt_fc.astype(float)
        mask[~lt_fc] = np.NaN
        # apply mask and get mean excluding NaNs, get moment
        omega0 = pd.Series(np.nanmean(disp.data * mask, axis=1), index=fc.index)
        density = sg.meta["density"]
        velocity = sg.meta["velocity"]
        moment = omega0 * 4 * np.pi * velocity ** 3 * density
        # calculate velocity_square_sum then normalize by frequency
        sample_spacing = np.median(vel.data.columns[1:] - vel.data.columns[:-1])
        # TODO this should work since we already divide by sqrt(N), check into it
        vel_sum = (vel.data ** 2).sum(axis=1) / sample_spacing
        energy = 4 * np.pi * vel_sum * density * velocity
        # create output source df and return
        df = pd.concat([omega0, fc, moment, energy], axis=1)
        cols = ["omega0", "fc", "moment", "energy"]
        names = ('method', 'parameter')
        df.columns = pd.MultiIndex.from_product([['maxmean'], cols], names=names)

        df[('maxmean', 'mw')] = (2/3) * np.log10(df[('maxmean', 'moment')]) - 6.0

        return self.new_from_dict({"source_df": self._add_to_source_df(df)})

    def _warn_on_missing_process(
            self, spreading=True, attenuation=True, radiation_pattern=True
    ):
        """ Issue warnings if various spectral corrections have not been issued. """
        base_msg = (
            f"calculating source parameters for {self} but "
            f"%s has not been corrected"
        )
        if radiation_pattern and not self.radiation_pattern_corrected:
            warnings.warn(base_msg % "radiation_pattern")
        if spreading and not self.spreading_corrected:
            warnings.warn(base_msg % "geometric spreading")
        if attenuation and not self.attenuation_corrected:
            warnings.warn(base_msg % "attenuation")
        return

    def _add_to_source_df(self, df):
        """
        Adds a dataframe of source parameters to the results dataframe.
        Returns a new dataframe. Will overwrite any existing columns with the
        same names.
        """
        current = self.source_df.copy()
        current[df.columns] = df
        # make sure the proper multi-index is set for columns
        if not isinstance(current.columns, pd.MultiIndex):
            current.columns = pd.MultiIndex.from_tuples(df.columns.values)
        return current

    # -------- Plotting functions

    def plot(
            self,
            event_id: Union[str, int],
            limit=None,
            stations: Optional[Union[str, int]] = None,
            show=True,
    ):
        """
        Plot a particular event id and scaled noise spectra.

        Parameters
        ----------
        event_id
            The event id (str) or event index (as stored in the SourceGroup).
            For example, 0 would return the first event stored in the df.
        stations
            The stations to plot
        limit
            If not None, only plot this many stations.
        show
            If False just return axis for future plotting.
        """
        from mopy.plotting import PlotEventSpectra

        event_spectra_plot = PlotEventSpectra(self, event_id, limit)
        return event_spectra_plot.show(show)

    def plot_centroid_shift(self, show=True, **kwargs):
        """
        Plot the centroid shift by distance differences for each event.
        """
        from mopy.plotting import PlotCentroidShift

        centroid_plot = PlotCentroidShift(self, **kwargs)
        return centroid_plot.show(show)

    def plot_time_domain(
            self,
            event_id: Union[str, int],
            limit=None,
            stations: Optional[Union[str, int]] = None,
            show=True,
    ):
        from mopy.plotting import PlotTimeDomain

        tdp = PlotTimeDomain(self, event_id, limit)
        return tdp.show(show)

    def plot_source_fit(
            self,
            event_id: Union[str, int],
            limit=None,
            stations: Optional[Union[str, int]] = None,
            show=True,
    ):
        from mopy.plotting import PlotSourceFit

        tdp = PlotSourceFit(self, event_id, limit)
        return tdp.show(show)

    # --- utils

    @property
    def spreading_corrected(self):
        """
        Return True if geometric spreading has been corrected.
        """
        return self.in_prococessing(self.correct_spreading.__name__)

    @property
    def attenuation_corrected(self):
        """
        Return True if attenuation has been corrected.
        """
        return self.in_prococessing(self.correct_attenuation.__name__)

    @property
    def radiation_pattern_corrected(self):
        """
        Return True if the radiation pattern has been corrected.
        """
        return self.in_prococessing(self.correct_radiation_pattern.__name__)




# df[('maxmean', 'mw')] = (2/3) * np.log10(df[('maxmean', 'moment')]) - 6.0
# dff = df['maxmean']
# dff['potency'] = dff['moment'].divide(self.meta['shear_modulus'], axis=0)
# dff['apparent_stress'] = dff['energy'] / dff['potency']
# import matplotlib.pyplot as plt
#
#
# plt.plot(np.log10(dff['potency']), np.log10(dff['energy']), '.')
#
# plt.show()
#
# breakpoint()
#
# apparent = dff['apparent_stress'].dropna()
#
#
#
# apparent = apparent[~np.isinf(apparent)]
#
#
#
# # apparent.hist()
# # plt.show()
# breakpoint()
#
#
#
# # add data frame to results