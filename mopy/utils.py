"""
Utility functions
"""
from __future__ import annotations

import functools
import importlib
import inspect
from collections import defaultdict
from types import ModuleType
from typing import Optional, Union, Mapping, Callable, Collection

import numpy as np
import obsplus
import obspy
import obspy.core.event as ev
import pandas as pd
from obsplus.constants import NSLC
from obspy.signal.invsim import corn_freq_2_paz
from obsplus.utils import get_reference_time, to_datetime64

from mopy.constants import MOTION_TYPES, PICK_COLS, AMP_COLS
from mopy.exceptions import DataQualityError
NAT = np.datetime64('NaT')


def get_phase_window_df(
    event: ev.Event,
    max_duration: Optional[Union[float, int, Mapping]] = None,
    min_duration: Optional[Union[float, int, Mapping]] = None,
    channel_codes: Optional[Union[Collection, pd.Series]] = None,
    buffer_ratio: Optional[float] = None,
) -> pd.DataFrame:
    """
    Return a dataframe of phase picks for the event. Does the following:

    1) Removes the rejected picks.
    2) Defines pick time windows using either
        a) a corresponding amplitude whose type matches the phase hint of the pick
           and has a time window
        b) the start of the phase to the arrival time plus min_duration.

    Parameters
    ----------
    event
        The seismic event
    max_duration
        The maximum duration (in seconds) of a phase. Can either be a scalar,
        or a mapping whose keys are seed_ids and values are applied to that
        specific channel.
    min_duration
        The minimum duration (in seconds) of a phase. Can either be a scalar,
        or a mapping whose keys are seed_ids and values are applied to that
        specific channel.
    channel_codes
        If provided, supplies the needed information to expand the dataframe
        to include an entry for each channel on a station for a given pick.
        For example, this is used a P pick that occurs on a HHZ channel will
        also have an entry on HHE and HHN (assuming they are in the list).
    buffer_ratio
        If not None, the ratio of the total duration of the phase windows
        which should be added to BOTH the start and end of the phase window.
    """
    reftime = to_datetime64(get_reference_time(event))

    def _getattr_or_none(attr, out_type=None):
        """ return a function for getting the defined attr or return None"""

        def _func(obj):
            out = getattr(obj, attr, None)
            if out_type:
                out = out_type(out)
            return out

        return _func

    def _get_earliest_s_time(df):
        return df[df.phase_hint == "S"].time.min()

    def _get_extrema_like_df(df, extrema_arg):
        """
        get min or max argument with the same length as df.
        This is done so each rows duration can be compared to some
        row specific value.
        """
        if isinstance(extrema_arg, (Mapping, pd.Series)):
            return df["seed_id"].map(extrema_arg)
        else:
            return np.ones(len(df)) * extrema_arg

    def _get_picks_df():
        """ Get the picks dataframe, remove picks flagged as rejected. """
        pdf = obsplus.picks_to_df(event)
        # remove rejected picks
        pdf = pdf[pdf.evaluation_status != "rejected"]
        # add seed_id column  # TODO change this to seed_id
        pdf["seed_id"] = obsplus.utils.get_nslc_series(pdf)
        # rename the resource_id column for later merging
        pdf.rename(columns={"resource_id": "pick_id"}, inplace=True)
        return pdf

    def _add_amplitudes(df):
        """ Add amplitudes to picks """
        # expected_cols = ["pick_id", "twindow_start", "twindow_end", "twindow_ref"]
        dtypes = {'pick_id': str, 'twindow_start': np.timedelta64, "twindow_end": np.datetime64,
                  'twindow_ref': np.datetime64}
        amp_df = pd.DataFrame(event.amplitudes)
        # convert all resource_ids to str
        for col in amp_df.columns:
            if col.endswith("id"):
                amp_df[col] = amp_df[col].astype(str)
        if amp_df.empty:  # no data, init empty df with expected cols
            amp_df = pd.DataFrame(columns=list(dtypes)).astype(dtypes)
        else:
            # merge picks/amps together and calculate windows
            tw = amp_df["time_window"]
            amp_df["twindow_start"] = tw.apply(_getattr_or_none("start"), outtype=np.timedelta64)
            amp_df["twindow_end"] = tw.apply(_getattr_or_none("end"), outtype=np.timedelta64)
            amp_df["twindow_ref"] = tw.apply(_getattr_or_none("reference", float), outtype=np.datetime64)
        # merge and return
        amp_df = amp_df[list(dtypes)]
        # merged = df.merge(amp_df, left_on="resource_id", right_on="pick_id", how="left")
        merged = df.merge(amp_df, left_on="pick_id", right_on="pick_id", how="outer")
        assert len(merged) == len(df)
        return _add_starttime_end(merged)

    def _add_starttime_end(df):
        """ Add the time window start and end """
        # fill references with start times of phases if empty
        df.loc[df["twindow_ref"].isnull(), "twindow_ref"] = df["time"]
        twindow_start = df['twindow_start'].fillna(np.timedelta64(0, 'ns')).astype('timedelta64[ns]')

        twindow_end = df['twindow_end'].fillna(np.timedelta64(0, 'ns')).astype('timedelta64[ns]')
        # Determine start/end times of phase windows
        df["starttime"] = df["twindow_ref"] - twindow_start
        df["endtime"] = df["twindow_ref"] + twindow_end
        # add travel time
        df["travel_time"] = df["time"] - reftime
        # get earliest s phase by station
        _sstart = df.groupby(list(NSLC[:2])).apply(_get_earliest_s_time)
        sstart = _sstart.rename("s_start").to_frame().reset_index()
        # merge back into pick_df, use either defined window or S phase, whichever
        # is smaller.
        dd2 = df.merge(sstart, on=["network", "station"], how="left")
        # get dataframe indices for P
        p_inds = df[df.phase_hint == "P"].index
        # make sure P end times don't exceed s start times
        endtime_or_s_start = dd2[["s_start", "endtime"]].min(axis=1, skipna=True)
        df.loc[p_inds, "endtime"] = endtime_or_s_start[p_inds]
        duration = abs(df["endtime"] - df["starttime"])
        # Make sure all value are under phase duration time, else truncate them
        if max_duration is not None:
            max_dur = _get_extrema_like_df(df, max_duration)
            larger_than_max = duration > max_dur
            df.loc[larger_than_max, "endtime"] = df["starttime"] + max_duration
        # Make sure all values are at least min_phase_duration, else expand them
        if min_duration is not None:
            breakpoint()
            min_dur = _get_extrema_like_df(df, min_duration)
            small_than_min = duration < min_dur
            df.loc[small_than_min, "endtime"] = df["starttime"] + min_dur
        # sanity checks
        assert (df["endtime"] >= df["starttime"]).all()
        assert not (df["starttime"].isnull()).any()
        return df

    def _duplicate_on_same_stations(df):  # What is the purpose for doing this???
        """ Duplicate all the entries for channels that are on the same station. """
        # make a dict of channel[:-1] and matching channels
        assert channel_codes is not None
        code_lest_1 = defaultdict(list)
        for code in channel_codes:
            code_lest_1[code[:-1]].append(code)
        # first create column to join on
        df["temp"] = df["seed_id"].str[:-1]
        # create expanded df
        new_inds = [x for y in df["seed_id"].unique() for x in code_lest_1[y[:-1]]]
        # get seed_id columns and merge back together
        df_new = pd.DataFrame(new_inds, columns=["seed_id"])
        seed_id = expand_seed_id(df_new["seed_id"])
        df_new = df_new.join(seed_id)
        # now merge in old dataframe for full expand
        df_new["temp"] = df_new["seed_id"].str[:-1]
        right_cols = list(PICK_COLS + AMP_COLS + ("phase_hint", "temp"))
        out = pd.merge(df_new, df[right_cols], on="temp", how="left")
        return out.drop_duplicates()

    def _apply_buffer(df):
        # add buffers on either end of waveform for tapering
        buff = (df["endtime"] - df["starttime"]) * buffer_ratio
        df["starttime"] = df["starttime"] - buff
        df["endtime"] = df["endtime"] + buff
        return df

    # read picks in and filter out rejected picks
    breakpoint()
    dd = _add_amplitudes(_get_picks_df())
    # return columns
    cols = list(NSLC + PICK_COLS + AMP_COLS + ("seed_id", "phase_hint"))
    out = dd[cols]
    # add buffer to window start/end
    if buffer_ratio is not None:
        out = _apply_buffer(out)
    # if channel codes are provided, make a duplicate of each phase window row
    # for each channel on the same station
    if channel_codes:
        out = _duplicate_on_same_stations(out)[cols]
    return out


# -------- Stream pre-processing


def _preprocess_node_stream(st: obspy.Stream) -> obspy.Stream:
    def _remove_node_response(st) -> obspy.Stream:
        """ using the fairfield files, remove the response through deconvolution """
        stt = st.copy()
        paz_5hz = corn_freq_2_paz(5.0, damp=0.707)
        paz_5hz["sensitivity"] = 76700
        pre_filt = (0.25, 0.5, 180.0, 200.0)
        stt.simulate(paz_remove=paz_5hz, pre_filt=pre_filt)
        return stt

    def _preprocess(st):
        """ Apply pre-processing """
        # detrend sort, remove response, set orientations
        stt = st.copy()
        stt.detrend("linear")
        stt.sort()
        return _remove_node_response(stt)

    return _preprocess(_remove_node_response(st))


def _get_phase_stream(st, ser, buffer=0.15):
    """
    Return the stream snipped out around phase. Pull more data than needed
    so a taper can be applied, after Oye et al. 2005.
    """
    # get starttime and endtime to pull from stream
    duration = ser["endtime"] - ser["starttime"]
    tbuff = duration * buffer
    t1 = obspy.UTCDateTime(ser["starttime"] - tbuff)
    t2 = obspy.UTCDateTime(ser["endtime"] + tbuff)
    # get seed_id codes
    network, station = ser.network, ser.station
    # slice out time frame and taper
    stt = st.slice(starttime=t1, endtime=t2)
    return stt.select(network=network, station=station)


def _pad_or_resample(trace, frequencies):
    """ """
    return trace


def _taper_stream(tr, taper_buffer):
    """ Taper the stream, return new stream """
    start = tr.stats.starttime
    end = tr.stats.endtime
    dur = end - start
    buffer = taper_buffer
    tr.taper(type="hann", max_percentage=0.05, max_length=dur * buffer)
    return tr


def _get_all_motion_types(tr, motion_type):
    """ Get a dict of all motion types. First detrend """
    assert motion_type == "velocity"
    tr.detrend("linear")
    # init copies for performing transformations
    acc = tr.copy()
    dis = tr.copy()
    # perform integrations/differentiations
    dis.integrate()
    dis.detrend("linear")
    acc.differentiate()
    acc.detrend("linear")
    return dict(displacement=dis, velocity=tr, acceleration=acc)


def _prefft(trace_dict, taper_buffer, freq_count):
    """ Apply prepossessing before fft. Namely, tapering and zero padding. """
    # first apply tapering
    for motion_type, tr in trace_dict.items():
        if taper_buffer:  # apply tapering
            trace_dict[motion_type] = _taper_stream(tr, taper_buffer)
        # apply zero padding if needed
        if freq_count and len(tr.data) / 2 < freq_count:
            new_ar = np.zeros(freq_count * 2)
            new_ar[: len(tr.data)] = tr.data
            tr.data = new_ar
    return trace_dict


def trace_to_spectrum_df(
    trace: obspy.Trace,
    motion_type: str,
    freq_count=None,
    taper_buffer=0.15,
    min_length=20,
    **kwargs,
) -> pd.DataFrame:
    """
    Convert a trace to a spectrum dataframe.

    The trace's response should have been removed, and the motion type must
    be provided.

    Parameters
    ----------
    trace
        Trace containing the data
    motion_type
        Either acceleration, velocity, or displacement
    freq_count
        If not None, the number of frequencies the dataframe have
        between 0 and the Nyquist frequency. If less than the nyquist it will
        be trimmed from the end. If greater than the nyquist it will be zero
        padded. The actual frequencies will be freq_count + 1 due to zero
        frequency.
    taper_buffer
        The amount to buffer each trace on each end.
    min_length
        The minimum number of samples that should be in the output df, else
        raise
    """
    assert motion_type == "velocity", "only velocity supported for now"
    # trim from beginning to freq_count * 2 if needed
    trace = trace.copy()  # don't mutate the data!
    if freq_count and freq_count < len(trace.data) / 2:
        trace.data = trace.data[: freq_count * 2]
    tr_dict = _get_all_motion_types(trace, motion_type=motion_type)
    tr_dict = _prefft(tr_dict, taper_buffer=taper_buffer, freq_count=freq_count)
    # get sampling rate and ensure all traces are the same length
    sr = tr_dict["velocity"].stats.sampling_rate
    lens = list({len(tr.data) for tr in tr_dict.values()})
    assert len(lens) == 1, "all traces should have the same length"
    # create dataframe and return
    ar = np.vstack([np.fft.rfft(x.data) for x in tr_dict.values()])
    freqs = np.fft.rfftfreq(lens[0], 1.0 / sr)
    df = pd.DataFrame(ar.T, index=freqs, columns=list(MOTION_TYPES))
    if len(df) < min_length:
        msg = (
            f"trace from {trace.id} is {len(df)} samples but {min_length}"
            f" are required!"
        )
        raise DataQualityError(msg)
    return df


# ------------- spectrum processing stuff


def _func_and_kwargs_str(func, *args, **kwargs):
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


def optional_import(module_name) -> ModuleType:
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
    return seed_id.str.split(".", expand=True).rename(columns=seed_id_map)


def pad_or_trim(array: np.ndarray, sample_count: int, pad_value: int = 0) -> np.ndarray:
    """
    Pad or trim an array to a specified length along the last dimension.

    Parameters
    ----------
    array
        A non-empty numpy array.
    sample_count
        The sample count to trim or pad to. If greater than the length of the
        arrays's last dimension the array will be padded with pad_value, else
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
