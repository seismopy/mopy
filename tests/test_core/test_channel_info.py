"""
Tests for channel information, specifically creating dataframes from it.
"""
import mopy
from mopy.constants import CHAN_COLS


class TestBasics:
    def test_type(self, node_channel_info):
        assert isinstance(node_channel_info, mopy.ChannelInfo)

    def test_has_traces(self, node_channel_info):
        """ Test meta df """
        data_df = node_channel_info.data
        assert set(CHAN_COLS).issubset(data_df.columns)

    def test_distance(self, node_channel_info):
        """ Test the channel info has reasonable distances. """
        df = node_channel_info.data
        dist = df['distance']
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
        assert df1.drop(columns='trace').equals(df2.drop(columns='trace'))
        # make sure traces werent copied
        tr_id1 = {id(tr) for tr in df1['trace']}
        tr_id2 = {id(tr) for tr in df2['trace']}
        assert tr_id1 == tr_id2
