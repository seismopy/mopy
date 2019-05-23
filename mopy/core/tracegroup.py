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

from mopy.core import DataFrameGroupBase, ChannelInfo
from mopy.utils import _source_process


class TraceGroup(DataFrameGroupBase):
    """
    Class for storing time series as pandas dataframes.

    Will also copy and update channel info.
    """

    def __init__(
        self,
        channel_info: ChannelInfo,
        waveforms: WaveformClient,
        motion_type: str,
        preprocess: Optional[Callable[[Stream], Stream]] = None,
    ):
        """
        Init a TraceGroup object.

        Parameters
        ----------
        channel_info
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
            waveforms into motion_type.
        """
        super().__init__()
        self.channel_info = channel_info.copy()
        self.stats = channel_info._stats
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
            values[i, 0 : len(stream[0].data)] = stream[0].data
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
