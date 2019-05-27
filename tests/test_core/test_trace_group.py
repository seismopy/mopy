"""
tests for the trace group
"""
import pytest
import numpy as np

import numpy as np

import mopy
import mopy.core.channelinfo
import mopy.core.tracegroup
from mopy import ChannelInfo, TraceGroup, SpectrumGroup
from mopy.exceptions import DataQualityError


class TestBasics:
    def test_type(self, node_trace_group):
        """ Ensure the type is correct. """
        assert isinstance(node_trace_group, mopy.core.tracegroup.TraceGroup)

    def test_dataframe_filled_with_nan(self, node_trace_group_raw):
        """ Ensure the dataframe is there and has Nulls"""
        df = node_trace_group_raw.data
        assert df.isnull().any().any()
        # each row should have some non-zero values
        assert (abs(df) > 0).any(axis=1).all()

    def test_filled_trace_group(self, node_trace_group):
        """ Ensure the dataframe was filled with 0s """
        df = node_trace_group.data
        assert not df.isnull().any().any()

    def test_missing_stream_warns(self, node_catalog, node_inventory, node_st):
        """
        Ensure a missing stream issues a warning.
        """
        # find the channel of the first pick and remove it from stream
        pick1 = node_catalog[0].picks[0]
        station_to_exclude = pick1.waveform_id.station_code
        st = node_st.copy()
        st.traces = [x for x in st if x.stats.station != station_to_exclude]
        assert len(st) < len(node_st)
        # create the channel info
        chaninfo = ChannelInfo(node_catalog, node_inventory)
        # A warning should be issued when it fails to find the station
        with pytest.warns(UserWarning):
            TraceGroup(chaninfo, st, motion_type="velocity")

    def test_nan_time_windows(self, node_channel_info, node_st):
        """
        Make sure can gracefully handle having one or more events with missing time windows
        """
        channel_info = node_channel_info.copy()
        # Clear the time windows
        channel_info.data.tw_start = np.nan
        channel_info.data.tw_end = np.nan
        # Try to create a TraceGroup with the NaN time windows
        # For now this is going to fail, but I think it should maybe issue a warning instead?
        with pytest.raises(ValueError):
            TraceGroup(channel_info, node_st, motion_type="velocity")


class TestToSpectrumGroup:
    """ Tests for converting the TraceGroup to SpectrumGroups. """

    @pytest.fixture
    def fft(self, node_trace_group):
        """ Convert the trace group to a spectrum group"""
        return node_trace_group.fft()

    @pytest.fixture
    def mtspec1(self, node_trace_group):
        """ Convert the trace group to a spectrum group via mtspec. """
        pytest.importorskip('mtspec')  # skip if mtspec is not installed
        return node_trace_group.mtspec()

    @pytest.fixture(params=('mtspec1', 'fft', ))
    def spec_from_trace(self, request):
        """ A gathering fixture for generic SpectrumGroup tests. """
        return request.getfixturevalue(request.param)

    # - General tests
    def test_type(self, spec_from_trace):
        """ Ensure the correct type was returned. """
        assert isinstance(spec_from_trace, SpectrumGroup)

    def test_parseval_theorem(self, node_trace_group, spec_from_trace):
        """
        The total power in the spectra should be roughly the same as in the
        time domain when scaled to number of samples.
        """
        meta = node_trace_group.meta
        df1 = abs(node_trace_group.data) ** 2
        df2 = abs(spec_from_trace.data) ** 2
        # scale time domain energy to number of samples
        norm = len(node_trace_group.data.columns) / meta['sample_count']
        sum1_scaled = df1.sum(axis=1) * norm
        # get freq. domain power (already scaled to number of samples)
        sum2 = df2.sum(axis=1)
        # the ratios **should** be about 1 (some are off though)
        # TODO see why some of these are off (up to 4) in Noise phase
        ratio = sum1_scaled / sum2
        assert abs(ratio.mean() - 1) < .1
        assert abs(ratio.median() - 1) < .1

    def test_compare_fft_mtspec(self, mtspec1, fft):
        """ fft and mtspec1 should not be radically different. """
        # calculate power sums
        df1, df2 = mtspec1.data, abs(fft.data)
        sum1 = (df1**2).sum()
        sum2 = (df2**2).sum()
        # compare ratios between fft and mtspec
        ratio = sum1 / sum2
        assert abs(ratio.mean() - 1.0) < 0.1
        assert abs(ratio.median() - 1.0) < 0.1
