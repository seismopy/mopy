"""
tests for the trace group
"""
import pytest

import mopy
from mopy.exceptions import DataQualityError


class TestBasics:
    def test_type(self, node_trace_group):
        """ Ensure the type is correct. """
        assert isinstance(node_trace_group, mopy.TraceGroup)

    def test_dataframe(self, node_trace_group):
        """ Ensure the dataframe is there and has no Nulls"""
        df = node_trace_group.data
        assert not df.isnull().any().any()
        # each row should have some non-zero values
        assert (abs(df) > 0).any(axis=1).all()

    def test_empty_stream_raises(self, node_catalog, node_inventory):
        """
        Ensure passing an empty catalog raises an excpetion.
        """
        empty_st_dict = {}
        with pytest.raises(DataQualityError):
            mopy.ChannelInfo(st_dict=empty_st_dict, catalog=node_catalog,
                             inventory=node_inventory)

    def test_missing_stream_warns(self, node_catalog, node_inventory,
                                  node_st_dict):
        """
        Ensure a missing stream issues a warning.
        """
        stdict = dict(node_st_dict)
        first_eid = str(node_catalog[0].resource_id)
        stdict.pop(first_eid)

        with pytest.warns(UserWarning):
            mopy.ChannelInfo(node_catalog.copy(), node_inventory, stdict)

