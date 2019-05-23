"""
tests for the trace group
"""
import pytest

import mopy
import mopy.core.channelinfo
import mopy.core.tracegroup
from mopy import ChannelInfo, TraceGroup
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
