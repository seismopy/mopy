"""
Utility functions
"""
from __future__ import annotations

import functools
import importlib
import inspect
from collections import defaultdict
from types import ModuleType
from typing import Optional, Union, Mapping, Callable, Collection, Hashable

import numpy as np
import obsplus
import obspy
import obspy.core.event as ev
import pandas as pd
from obsplus.constants import NSLC
from obspy.signal.invsim import corn_freq_2_paz
from obsplus.utils import get_reference_time, to_datetime64, to_timedelta64

from mopy.constants import MOTION_TYPES, PHASE_WINDOW_INTERMEDIATE_COLS, PHASE_WINDOW_DF_COLS
from mopy.exceptions import DataQualityError, NoPhaseInformationError

NAT = np.datetime64('NaT')


def get_phase_window_df(
    event: ev.Event,
    max_duration: Optional[Union[float, int, Mapping]] = None,
    min_duration: Optional[Union[float, int, Mapping]] = None,
    channel_codes: Optional[Union[Collection, pd.Series]] = None,
    buffer_ratio: Optional[float] = None,
    restrict_to_arrivals: bool = True,
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
    restrict_to_arrivals
        If True, only use picks for which there is an arrival on the preferred
        origin.
    """
    reftime = to_datetime64(get_reference_time(event))

    def _getattr_or_none(attr):
        """ return a function for getting the defined attr or return None"""

        def _func(obj, out_type=None):
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

        if isinstance(extrema_arg, pd.Series):
            return df["seed_id"].map(extrema_arg.droplevel("seed_id_less"))
        elif isinstance(extrema_arg, Mapping):
            return df["seed_id"].map(extrema_arg)
        else:
            return np.ones(len(df)) * extrema_arg

    def _get_picks_df(restrict_to_arrivals):
        """ Get the picks dataframe, remove picks flagged as rejected. """
        pdf = obsplus.picks_to_df(event)
        pdf["seed_id_less"] = pdf["seed_id"].str[:-1]
        if restrict_to_arrivals:
            adf = obsplus.arrivals_to_df(event)
            pdf = pdf.loc[pdf["resource_id"].isin(adf["pick_id"])]
        # remove rejected picks
        pdf = pdf[pdf.evaluation_status != "rejected"]
        # TODO: There are three (four?) options for the proper way to handle this, and I'm not sure which is best:
        #  1. Validate the event and raise if there are any S-picks < P-picks
        #  2. Repeat the above, but just silently skip the event
        #  3. Repeat the above, but drop any picks that are problematic
        #  4. Repeat the above, but have a separate flag to indicate whether to drop the picks or forge ahead
        #  Also, I know there is a validator in obsplus that will check these, but I dunno about removing the offending picks...
        #  Ideally, at least for my purposes, I'm going to fix the underlying issue with my location code and this will be moot
        if {"P", "S"}.issubset(pdf["phase_hint"]):
            phs = pdf.groupby("phase_hint")
            p_picks = phs.get_group("P")
            s_picks = phs.get_group("S")
            both = set(p_picks["seed_id_less"]).intersection(s_picks["seed_id_less"])
            p_picks = p_picks.loc[p_picks["seed_id_less"].isin(both)].set_index("seed_id_less").sort_index()
            s_picks = s_picks.loc[s_picks["seed_id_less"].isin(both)].set_index("seed_id_less").sort_index()
            bad_s = s_picks.loc[p_picks["time"] > s_picks["time"]]
            pdf = pdf.loc[~pdf["resource_id"].isin(bad_s["resource_id"])]
        if not len(pdf):
            raise NoPhaseInformationError(f"No valid phases for event:\n{event}")
        # # add seed_id column
        # pdf["seed_id"] = obsplus.utils.get_nslc_series(pdf)
        # add the seed id column that drops the component from the channel
        # rename the resource_id column for later merging
        pdf.rename(columns={"resource_id": "pick_id"}, inplace=True)
        return pdf

    def _add_amplitudes(df):
        """ Add amplitudes to picks """
        # expected_cols = ["pick_id", "twindow_start", "twindow_end", "twindow_ref"]
        dtypes = {'pick_id': str, 'twindow_start': np.timedelta64, "twindow_end": np.timedelta64,
                  'twindow_ref': np.datetime64}
        amp_df = event.amplitudes_to_df()
        # Drop rejected amplitudes
        amp_df = amp_df.loc[amp_df["evaluation_status"] != "rejected"]
        if amp_df.empty:  # no data, init empty df with expected cols
            amp_df = pd.DataFrame(columns=list(dtypes)).astype(dtypes)
        else:
            # # convert all resource_ids to str  <- This should be unnecessary, because the to_df methods only ever store ids as str, no?
            # for col in amp_df.columns:
            #     if col.endswith("id"):
            #         amp_df[col] = amp_df[col].astype(str)
            # merge picks/amps together and calculate windows
            amp_df.rename(columns={"time_begin": "twindow_start", "time_end": "twindow_end", "reference": "twindow_ref"}, inplace=True)
        # merge and return
        amp_df = amp_df[list(dtypes)]
        # Note: the amplitude list can be longer than the pick list because of
        # the logic for dropping picks earlier
        merged = df.merge(amp_df, left_on="pick_id", right_on="pick_id", how="outer").dropna(subset=["time"])
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
        _s_start = df.groupby(list(NSLC[:2])).apply(_get_earliest_s_time)
        s_start = _s_start.rename("s_start").to_frame().reset_index()
        # merge back into pick_df, use either defined window or S phase, whichever
        # is smaller.
        dd2 = df.merge(s_start, on=["network", "station"], how="left")
        # get dataframe indices for P
        p_inds = df[df.phase_hint == "P"].index
        # make sure P end times don't exceed s start times
        endtime_or_s_start = dd2[["s_start", "endtime"]].min(axis=1, skipna=True)
        df.loc[p_inds, "endtime"] = endtime_or_s_start[p_inds]
        duration = abs(df["endtime"] - df["starttime"])
        # Make sure all value are under phase duration time, else truncate them
        if max_duration is not None:
            max_dur = to_timedelta64(_get_extrema_like_df(df, max_duration))
            larger_than_max = duration > max_dur
            df.loc[larger_than_max, "endtime"] = df["starttime"] + to_timedelta64(max_duration)
        # Make sure all values are at least min_phase_duration, else expand them
        if min_duration is not None:
            min_dur = to_timedelta64(_get_extrema_like_df(df, min_duration))
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
            code_lest_1[code[0]].append(code[1])
        # create expanded df
        new_inds = [x for y in df["seed_id"].unique() for x in code_lest_1[y[:-1]]]
        # get seed_id columns and merge back together
        df_new = pd.DataFrame(new_inds, columns=["seed_id"])
        df_new["seed_id_less"] = df_new["seed_id"].str[:-1]
        seed_id = expand_seed_id(df_new["seed_id"])
        df_new = df_new.join(seed_id)
        # now merge in old dataframe for full expand
        # df_new["temp"] = df_new["seed_id"].str[:-1]
        right_cols = list(PHASE_WINDOW_INTERMEDIATE_COLS)
        out = pd.merge(df_new, df[right_cols], on="seed_id_less", how="left")
        return out.drop_duplicates()

    def _apply_buffer(df):
        # add buffers on either end of waveform for tapering
        buff = (df["endtime"] - df["starttime"]) * buffer_ratio
        df["starttime"] = df["starttime"] - buff
        df["endtime"] = df["endtime"] + buff
        return df

    # read picks in and filter out rejected picks
    dd = _add_amplitudes(_get_picks_df(restrict_to_arrivals))
    # return columns
    out = dd[list(PHASE_WINDOW_DF_COLS)]
    # add buffer to window start/end
    if buffer_ratio is not None:
        out = _apply_buffer(out)
    # if channel codes are provided, make a duplicate of each phase window row
    # for each channel on the same station
    if channel_codes:
        out = _duplicate_on_same_stations(out)[list(PHASE_WINDOW_DF_COLS)]
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


def fill_column(df: pd.DataFrame, col_name: Hashable, fill: Union[pd.Series, Mapping, str, int, float], na_only: bool = True) -> None:
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
        df[col_name] = fill
    else:
        df[col_name].fillna(fill, inplace=True)


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
