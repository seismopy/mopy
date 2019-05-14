"""
Time domain rep. of waveforms in mopy.
"""
import numpy as np
import obspy
import pandas as pd
from scipy.fftpack import next_fast_len

from mopy.config import get_default_param
from mopy.core import DataFrameGroupBase, ChannelInfo


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