"""
Time domain rep. of waveforms in mopy.
"""
import warnings
from functools import lru_cache
from typing import Optional, Callable

import numpy as np
import obspy
import pandas as pd
from obsplus.constants import NSLC
from obsplus.interfaces import WaveformClient
from obsplus.waveforms.utils import stream_bulk_split
from obspy import Stream
from scipy.fftpack import next_fast_len

import mopy
from mopy.core import DataFrameGroupBase, StatsGroup
from mopy.utils import _source_process, optional_import, pad_or_trim


class TraceGroup(DataFrameGroupBase):
    """
    Class for storing time series as pandas dataframes.

    Will also copy and update channel info.
    """

    def __init__(
            self,
            stats: StatsGroup,
            waveforms: WaveformClient,
            motion_type: str,
            preprocess: Optional[Callable[[Stream], Stream]] = None,
    ):
        """
        Init a TraceGroup object.

        Parameters
        ----------
        stats
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
        super().__init__()
        self.channel_info = stats.copy()
        self._stats_group = stats
        self.stats.motion_type = motion_type
        # get an array of streams
        st_array = self._get_st_array(waveforms, preprocess)
        self.data = self._make_trace_df(st_array)

    def _make_trace_df(self, st_array):
        """
        Make the dataframe containing time series data.
        """
        sampling_rate = self.sampling_rate

        # figure out with streams are fractured and drop them
        good_st, new_ind = self._filter_stream_array(st_array)

        # get lens and create array empty array with next fast fft length
        lens = [len(x[0]) for x in good_st]
        max_fast = next_fast_len(max(lens) + 1)
        values = np.empty((len(new_ind), max_fast)) * np.NaN
        # iterate each stream and fill array
        for i, stream in enumerate(good_st):
            values[i, 0: len(stream[0].data)] = stream[0].data
        # init df from filled values
        time = np.arange(0, float(max_fast) / sampling_rate, 1.0 / sampling_rate)
        df = pd.DataFrame(values, index=new_ind, columns=time)
        # set name of columns
        df.columns.name = "time"
        # add original lengths to the channel_info
        self.channel_info.data.loc[new_ind, "sample_count"] = lens
        return df

    def _filter_stream_array(self, st_array):
        """ Filter the stream array, issue warnings if quality is not met. """
        phase_df = self.channel_info.data
        # determine which streams have contiguous data, issue warnings otherwise
        is_good = np.array([len(x) == 1 for x in st_array])
        new_ind = phase_df.index[is_good]
        # issue warnings about any data fetching failures
        bad_ind = phase_df.index[~is_good]
        if len(bad_ind):
            msg = "failed to get waveforms for the following\n: {tuple(bad_ind}"
            warnings.warn(msg)
        return st_array[is_good], new_ind

    def _get_st_array(self, waveforms, preprocess):
        """ Return an array of streams, one for each row in chan info. """
        phase_df = self.channel_info.data
        if (phase_df["tw_start"].isnull() | phase_df["tw_end"].isnull()).any():
            raise ValueError(
                "Time windows must be assigned to the ChannelInfo prior to TraceGroup creation"
            )
        # get bulk argument and streams
        bulk = self._get_bulk(phase_df)
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

    def _get_bulk(self, phase_df):
        """ Get bulk request from channel_info df """
        ser = phase_df[["tw_start", "tw_end"]].reset_index()
        nslc = ser["seed_id"].str.split(".", expand=True)
        nslc.columns = list(NSLC)
        df = nslc.join(ser[["tw_start", "tw_end"]])
        df["tw_start"] = df["tw_start"].apply(obspy.UTCDateTime)
        df["tw_end"] = df["tw_end"].apply(obspy.UTCDateTime)
        return df.to_records(index=False).tolist()

    def fft(self, sample_count: Optional[int] = None) -> "mopy.SpectrumGroup":
        """
        Return a SpectrumGroup by applying fft.

        Parameters
        ----------
        sample_count
            The number of samples in the time series to use in the
            transformation. If greater than the length of the data columns
            it will be zero padded, if less the data will be cropped.
            Defaults to the next fast length.
        """
        data = self.data
        if sample_count is None:
            sample_count = next_fast_len(len(self.data.columns))
        spec = np.fft.rfft(data, n=sample_count, axis=-1)
        # increase all but zero freq. to account for rfft missing neg. freqs.
        # note sqrt of 2, rather than 2, is used because power should double
        # not amplitude.
        spec[1:] = spec[1:] * np.sqrt(2)
        freqs = np.fft.rfftfreq(len(data.columns), 1.0 / self.sampling_rate)
        df = pd.DataFrame(spec, index=data.index, columns=freqs)
        df.columns.name = "frequency"
        # normalize for DFT
        df = df.divide(np.sqrt(self.stats["sample_count"]), axis=0)
        # create spec group
        kwargs = dict(data=df, channel_info=self.channel_info, stats=self.stats)
        return mopy.SpectrumGroup(**kwargs)

    def mtspec(
            self, time_bandwidth=2, sample_count: Optional[int] = None, **kwargs
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
        df.columns.name = "frequency"
        # convert from PSD to amplitude spectra
        df *= self.sampling_rate  # multiply by SR to de-densify
        df = np.sqrt(df)  # power to amplitude
        # normalize to number of non-zero samples
        norm = np.sqrt(len(self.data.columns)) / np.sqrt(self.stats["sample_count"])
        df = df.multiply(norm, axis=0)
        # collect and return
        sg_kwargs = dict(data=df, channel_info=self.channel_info, stats=self.stats)
        return mopy.SpectrumGroup(**sg_kwargs)

    @property
    @lru_cache()
    def sampling_rate(self):
        """ return the sampling rate of the data. """
        # get sampling rate. Currently only one sampling rate is supported.
        sampling_rates = self.channel_info.data["sampling_rate"].unique()
        assert len(sampling_rates) == 1
        return sampling_rates[0]

    # -------- Methods

    @_source_process
    def fillna(self, fill_value=0) -> "TraceGroup":
        """
        Fill any NaN with fill_value.

        Parameters
        ----------
        fill_value
        """
        return self.new_from_dict({"data": self.data.fillna(fill_value)})
