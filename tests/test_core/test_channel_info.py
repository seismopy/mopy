"""
Tests for channel information, specifically creating dataframes from it.
"""
import pandas as pd

import mopy
import mopy.core.channelinfo
from mopy.constants import CHAN_COLS


class TestBasics:
    def test_type(self, node_channel_info):
        assert isinstance(node_channel_info, mopy.core.channelinfo.ChannelInfo)

    def test_has_traces(self, node_channel_info):
        """ Test meta df """
        data_df = node_channel_info.data
        assert set(CHAN_COLS).issubset(data_df.columns)

    def test_distance(self, node_channel_info):
        """ Test the channel info has reasonable distances. """
        df = node_channel_info.data
        dist = df["distance"]
        assert not dist.isnull().any()
        assert (dist > 0).all()

    def test_copy(self, node_channel_info):
        """ Ensure copying doesnt copy traces. """
        cop = node_channel_info.copy()
        # the base objects should have been copied
        assert id(cop) != id(node_channel_info)
        assert id(cop.data) != id(node_channel_info.data)
        # make sure the values are equal
        df1 = node_channel_info.data
        df2 = cop.data
        assert df1.equals(df2)

    def test_add_time_buffer(self, node_channel_info):
        """
        Ensure time can be added to the start and enf of the node_trace_group.
        """
        # Add times, start and end
        df = node_channel_info.data
        start = 1
        end = pd.Series(2, index=df.index)
        tg = node_channel_info.add_time_buffer(start=start, end=end)
        # Make sure a copy did occur
        assert tg is not node_channel_info
        # Make sure time offset is correct
        df2 = tg.data
        assert ((df2["tw_start"] + 1) == df["tw_start"]).all()
        assert ((df2["tw_end"] - 2) == df["tw_end"]).all()
