"""
Time domain rep. of waveforms in mopy.
"""
from __future__ import annotations

import warnings
from typing import Optional, Callable, Tuple, List

import numpy as np
import obspy
import pandas as pd
from obsplus.constants import NSLC, waveform_clientable_type
from obsplus.interfaces import WaveformClient
from obsplus.utils.waveforms import stream_bulk_split
from obspy import Stream
from scipy.fftpack import next_fast_len

import mopy
from mopy.core.base import DataGroupBase
from mopy.core import StatsGroup
from mopy.utils.misc import _track_method, optional_import, pad_or_trim
from mopy.exceptions import NoPhaseInformationError


# This class will be pushed to ObsPlus someday; don't use it directly
class _TraceGroup(DataGroupBase):
    """
    Class for storing time series as pandas dataframes.

    Will also copy and update channel info.
    """

    def __init__(
        self,
        stats_group: StatsGroup,
        waveforms: WaveformClient,
        motion_type: str,
        preprocess: Optional[Callable[[Stream], Stream]] = None,
    ):
        """
        Init a TraceGroup object.

        Parameters
        ----------
        stats_group
            Contains all the meta information about the channel traces.
        waveforms
            Any object with a get_waveforms_bulk method. It is recommended you
            use an obspy.Stream object which has already been preprocessed.
        motion_type
            The type of ground motion returned by waveforms.
        preprocess
            If not None, a callable that takes a stream as an argument and
            returns a pre-processed stream. This function should at least
            remove the instrument response and detrend. It should put all
            waveforms into motion_type (acceleration, velocity or displacement).
        """
        if not len(stats_group):
            raise NoPhaseInformationError(
                "StatsGroup does not have any pick information"
            )
        sg_with_motion = stats_group.add_columns(motion_type=motion_type)
        super().__init__(sg_with_motion)
        # get an array of streams
        st_array = self._get_st_array(waveforms, preprocess)
        self.data = self._make_trace_df(st_array)

    def _make_trace_df(self, st_array: np.ndarray) -> pd.DataFrame:
        """
        Make the dataframe containing time series data.
        """
        sampling_rate = self.sampling_rate
        assert sampling_rate is not None

        # figure out with streams are fractured and drop them
        good_st, new_ind = self._filter_stream_array(st_array)

        # get lens and create array empty array with next fast fft length
        lens = [len(x[0]) for x in good_st]
        max_fast = next_fast_len(max(lens))  # + 1)  # What is the point of the +1?
        values = np.empty((len(new_ind), max_fast)) * np.NaN
        # iterate each stream and fill array
        for i, stream in enumerate(good_st):
            values[i, 0 : len(stream[0].data)] = stream[0].data
        # init df from filled values
        time = np.arange(0, float(max_fast) / sampling_rate, 1.0 / sampling_rate)
        df = pd.DataFrame(values, index=new_ind, columns=time)
        # set name of columns
        df.columns.name = "time"
        # add original lengths to the channel_info
        self.stats.loc[new_ind, "npts"] = lens
        return df

    def _filter_stream_array(self, st_array: np.ndarray) -> Tuple[np.ndarray, pd.Index]:
        """ Filter the stream array, issue warnings if quality is not met. """
        # determine which streams have contiguous data, issue warnings otherwise
        starttimes = np.array(
            [x[0].stats.starttime.timestamp if len(x) else np.nan for x in st_array]
        )
        endtimes = np.array(
            [x[0].stats.endtime.timestamp if len(x) else np.nan for x in st_array]
        )
        stats_start = (
            (self.stats["starttime"] - pd.Timestamp("1970-01-01"))
            // pd.Timedelta("1ns")
            / np.float64(1e9)
        )
        stats_end = (
            (self.stats["endtime"] - pd.Timestamp("1970-01-01"))
            // pd.Timedelta("1ns")
            / np.float64(1e9)
        )
        is_good = np.isclose(starttimes, stats_start, rtol=1e-12) & np.isclose(
            endtimes, stats_end, rtol=1e-12
        )

        new_ind = self.stats.index[is_good]
        # issue warnings about any data fetching failures
        bad_ind = self.stats.index[~is_good]
        if len(bad_ind):
            msg = f"failed to get waveforms for the following\n: {tuple(bad_ind)}"
            warnings.warn(msg)
        return st_array[is_good], new_ind

    def _get_st_array(
        self, waveforms: waveform_clientable_type, preprocess: Callable
    ) -> np.ndarray:
        """ Return an array of streams, one for each row in chan info. """
        stats = self.stats
        if (stats["starttime"].isnull() | stats["endtime"].isnull()).any():
            raise ValueError(
                "Time windows must be assigned to the StatsGroup prior to TraceGroup creation"
            )
        # get bulk argument and streams
        bulk = self._get_bulk(stats)
        # ensure waveforms is a stream, then get a list of streams
        if not isinstance(waveforms, obspy.Stream):
            waveforms = waveforms.get_waveforms_bulk(bulk)
        # apply preprocessor if provided
        if preprocess is not None:
            waveforms = preprocess(waveforms)
        # get array of streams
        st_list = stream_bulk_split(waveforms, bulk)
        ar = np.zeros(len(st_list)).astype(object)
        ar[:] = st_list
        return ar

    def _get_bulk(self, phase_df: pd.DataFrame) -> List:
        """ Get bulk request from channel_info df """
        ser = phase_df[["starttime", "endtime"]].reset_index()
        nslc = ser["seed_id"].str.split(".", expand=True)
        nslc.columns = list(NSLC)
        df = nslc.join(ser[["starttime", "endtime"]])
        df["starttime"] = df["starttime"].apply(obspy.UTCDateTime)
        df["endtime"] = df["endtime"].apply(obspy.UTCDateTime)
        return df.to_records(index=False).tolist()

    def dft(self, sample_count: Optional[int] = None) -> "mopy.SpectrumGroup":
        """
        Apply a discrete Fourier transform to the data to create a SpectrumGroup

        Parameters
        ----------
        sample_count
            The number of samples in the time series to use in the
            transformation. If greater than the length of the data columns
            it will be zero padded, if less the data will be cropped.
        """
        data = self.data
        if sample_count is None:
            sample_count = next_fast_len(len(self.data.columns))
        spec = np.fft.rfft(data.values, n=sample_count, axis=-1)
        freqs = np.fft.rfftfreq(len(data.columns), 1.0 / self.sampling_rate)
        df = pd.DataFrame(spec, index=data.index, columns=freqs)
        df.columns.name = "frequency"
        # create SpectrumGroup
        kwargs = dict(data=df, stats_group=self.stats_group, spectra_type="dft")
        return mopy.SpectrumGroup(**kwargs)

    def mtspec(
        self,
        time_bandwidth=2,
        sample_count: Optional[int] = None,
        to_dft: bool = False,
        **kwargs,
    ) -> "mopy.SpectrumGroup":
        """
        Return a SpectrumGroup by calculating amplitude spectra via mtspec.

        Parameters
        ----------
        time_bandwidth
            The time bandwidth used in mtspec, see its docstring for more
            details.
        sample_count
            The number of samples in the time series to use in the
            transformation. If greater than the length of the data columns
            it will be zero padded, if less the data will be cropped.
            Defaults to the next fast length.
        to_dft
            If True, convert the power spectral density calculated by mtspec to
            the discrete fourier transform

        Notes
        -----
        The parameter time_bandwidth and the kwargs are passed directly to
        mtspec.mtspec, see its documentation for details:
        https://krischer.github.io/mtspec/multitaper_mtspec.html
        """
        mtspec = optional_import("mtspec")
        # get prepared input array
        if sample_count is None:
            sample_count = next_fast_len(len(self.data.columns))
        ar = pad_or_trim(self.data.values, sample_count=sample_count)
        # collect kwargs for mtspec
        delta = 1.0 / self.sampling_rate
        kwargs = dict(kwargs)
        kwargs.update({"delta": delta, "time_bandwidth": time_bandwidth})
        # unfortunately we may need to use loops here
        out = [mtspec.mtspec(x, **kwargs) for x in ar]
        array = np.array([x[0] for x in out])
        freqs = out[0][1]
        df = pd.DataFrame(array, index=self.data.index, columns=freqs)
        # normalize to number of non-zero samples
        norm = len(self.data.columns) / self.stats["npts"]
        df = df.multiply(norm, axis=0)
        df.columns.name = "frequency"
        # # collect and return
        sg_kwargs = dict(data=df, stats_group=self.stats_group, spectra_type="psd")
        sg = mopy.SpectrumGroup(**sg_kwargs)
        if to_dft:
            sg = sg.to_spectra_type("dft")
        return sg

    # -------- Methods

    @_track_method
    def fillna(self, fill_value=0) -> "TraceGroup":
        """
        Fill any NaN with fill_value.

        Parameters
        ----------
        fill_value
        """
        return self.new_from_dict(data=self.data.fillna(fill_value))


class TraceGroup(_TraceGroup):
    """
    Class for storing time series as pandas dataframes.

    Will also copy and update channel info.
    """
