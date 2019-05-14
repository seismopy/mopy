"""
Utility functions
"""
import copy
import functools
import inspect
import warnings
from collections import defaultdict
from typing import Optional, Union, Mapping, Callable, Collection

import matplotlib.pyplot as plt
import numpy as np
import obsplus
import obspy
import obspy.core.event as ev
import pandas as pd
from decorator import decorator
from obsplus.constants import NSLC
from obspy.signal.invsim import corn_freq_2_paz

import mopy
from mopy.constants import MOTION_TYPES
from mopy.exceptions import DataQualityError


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
    reftime = obsplus.utils.get_reference_time(event)

    def _getattr_or_none(attr, out_type=None):
        """ return a function for getting the defined attr or return None"""

        def _func(obj):
            out = getattr(obj, attr, None)
            if out_type:
                out = out_type(out)
            return out

        return _func

    def _get_earlies_s_time(df):
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
        return pdf

    def _add_amplitudes(df):
        """ Add amplitudes to picks """
        expected_cols = ["pick_id", "twindow_start", "twindow_end", "twindow_ref"]
        amp_df = pd.DataFrame(event.amplitudes)
        # convert all resource_ids to str
        for col in amp_df.columns:
            if col.endswith("id"):
                amp_df[col] = amp_df[col].astype(str)
        if amp_df.empty:  # no data, init empty df with expected cols
            amp_df = pd.DataFrame(columns=expected_cols)
        else:
            # merge picks/amps together and calculate windows
            tw = amp_df["time_window"]
            amp_df["twindow_start"] = tw.apply(_getattr_or_none("start"))
            amp_df["twindow_end"] = tw.apply(_getattr_or_none("end"))
            amp_df["twindow_ref"] = tw.apply(_getattr_or_none("reference", float))
        # merge and return
        amp_df = amp_df[expected_cols]
        merged = df.merge(amp_df, left_on="resource_id", right_on="pick_id", how="left")
        assert len(merged) == len(df)
        return _add_tw_start_end(merged)

    def _add_tw_start_end(df):
        """ Add the time window start and end """
        # fill references with start times of phases if empty
        df.loc[df["twindow_ref"].isnull(), "twindow_ref"] = df["time"]
        # Determine start/end times of phase windows
        df["tw_start"] = df["twindow_ref"] - df["twindow_start"].fillna(0)
        df["tw_end"] = df["twindow_ref"] + df["twindow_end"].fillna(0)
        # add travel time
        df['travel_time'] = df['time'] - reftime.timestamp
        # get earliest s phase by station
        _sstart = df.groupby(list(NSLC[:2])).apply(_get_earlies_s_time)
        sstart = _sstart.rename("s_start").to_frame().reset_index()
        # merge back into pick_df, use either defined window or S phase, whichever
        # is smaller.
        dd2 = df.merge(sstart, on=["network", "station"], how="left")
        # get dataframe indicies for P
        p_inds = df[df.phase_hint == "P"].index
        # make sure P end times don't exceed s start times
        tw_end_or_s_start = dd2[["s_start", "tw_end"]].min(axis=1, skipna=True)
        df.loc[p_inds, "tw_end"] = tw_end_or_s_start[p_inds]
        duration = abs(df["tw_end"] - df["tw_start"])
        # Make sure all value are under phase duration time, else truncate them
        if max_duration is not None:
            max_dur = _get_extrema_like_df(df, max_duration)
            larger_than_max = duration > max_dur
            df.loc[larger_than_max, "tw_end"] = df["tw_start"] + max_duration
        # Make sure all values are at least min_phase_duration, else expand them
        if min_duration is not None:
            min_dur = _get_extrema_like_df(df, min_duration)
            small_than_min = duration < min_dur
            df.loc[small_than_min, "tw_end"] = df["tw_start"] + min_dur
        # sanity checks
        assert (df["tw_end"] >= df["tw_start"]).all()
        assert not (df["tw_start"].isnull()).any()
        return df

    def _duplicate_on_same_stations(df):
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
        seed_id_map = {num: code for num, code in enumerate(NSLC)}
        seed_id = df_new["seed_id"].str.split(".", expand=True).rename(columns=seed_id_map)
        df_new = df_new.join(seed_id)
        # now merge in old dataframe for full expand
        df_new["temp"] = df_new["seed_id"].str[:-1]
        right_cols = ["tw_start", "tw_end", "phase_hint", "time", "temp"]
        out = pd.merge(df_new, df[right_cols], on="temp", how="left")
        return out.drop_duplicates()

    def _apply_buffer(df):
        # add buffers on either end of waveform for tapering
        buff = (df["tw_end"] - df["tw_start"]) * buffer_ratio
        df["tw_start"] = df["tw_start"] - buff
        df["tw_end"] = df["tw_end"] + buff
        return df

    # read picks in and filter out rejected picks
    dd = _add_amplitudes(_get_picks_df())
    # return columns
    cols2keep_picks = list(NSLC) + ["phase_hint", "time"]
    cols2keep_amps = ["tw_start", "tw_end", "seed_id"]
    cols = cols2keep_amps + cols2keep_picks
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
    duration = ser.tw_end - ser.tw_start
    tbuff = duration * buffer
    t1 = obspy.UTCDateTime(ser.tw_start - tbuff)
    t2 = obspy.UTCDateTime(ser.tw_end + tbuff)
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
    trace = trace.copy()  # dont mutate the data!
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


@decorator
def _add_processing_info(func, *args, **kwargs):
    """
    This is a decorator that attaches information about a processing call as a
    string to an output Spectrum object
    """

    def _add_processing_info(result, info):
        if not getattr(result.stats, "processing", None):
            result.stats.processing = []
        result.stats.processing.append(info)

    info = _func_and_kwargs_str(func, *args, **kwargs)
    result = func(*args, **kwargs)
    # Attach after executing the function to avoid having it attached
    # while the operation failed.
    _add_processing_info(result, info)
    return result


@decorator
def _idempotent(func, *args, **kwargs):
    """
    Idempotent functions should only be run once.

    This decorator checks if the function has already been run, and just
    return self if so. If not, it will add the function name to a
    stats._idempotent list.
    """
    id_name = "_idempotent"
    func_name = func.__name__
    self = args[0]
    if func_name in getattr(self.stats, id_name, {}):
        msg = f"{func_name} has already been called on {self}, returning self"
        warnings.warn(msg)
        return self
    out = func(*args, **kwargs)
    # add func name
    ip_set = getattr(out.stats, id_name, set())
    ip_set.add(func_name)
    setattr(out.stats, id_name, ip_set)
    return out


def _source_process(idempotent: Union[Callable, bool] = False):
    """
    Mark a method as a source process.

    Each of source process method returns a copy of the source with the
    __dict__ updated with the output of the function.
    """

    def _deco(func):
        @functools.wraps(func)
        def _wrap(self, *args, **kwargs):
            info_str = _func_and_kwargs_str(func, self, *args, **kwargs)
            new = func(self, *args, **kwargs)
            if "processing" not in new.stats:
                new.stats.processing = ()
            new.stats.processing = tuple(list(new.stats.processing) + [info_str])
            # Makes sure the modify meta is called to update meta dataframe.
            if hasattr(new, "update_meta"):
                new.post_source_function_hook()

            return new

        return _wrap

    # this allows the decorator to be used with or without calling it.
    if callable(idempotent):
        wrapped_func = idempotent
        idempotent = False
        return _deco(wrapped_func)
    else:
        return _deco


def plot_spectrum(show=True, motion_type=None, **kwargs):
    """
    Plot any number of spectrum.

    Parameters
    ----------
    show
    motion_type
    kwargs
    """
    breakpoint()
    for label, spec in kwargs.items():
        if not isinstance(spec, mopy.EventSpectrum):
            continue
        spec.plot_amplitude(motion_type=motion_type, show=False, label=label)
    if show:
        plt.show()
