"""
Tests for functionality of parent Group classes.
"""
import pytest
from copy import deepcopy

import numpy as np

import mopy


class TestNewFromDict:
    """ Tests for inplace keyword. """

    def test_copy(self, node_stats_group):
        """ Ensure copying doesnt copy traces. """
        cop = node_stats_group.copy()
        # the base objects should have been copied
        assert id(cop) != id(node_stats_group)
        assert id(cop.data) != id(node_stats_group.data)

    def test_new_trace_group(self, node_trace_group):
        """ ensure a new trace group is returned """
        sg = node_trace_group.stats_group
        tg = node_trace_group.new_from_dict()
        assert tg is not node_trace_group
        assert sg is not tg.stats_group

    def test_inplace(self, node_trace_group):
        """ ensure inplace does not make a copy. """
        tg = node_trace_group.copy()
        tg2 = tg.new_from_dict(inplace=True)
        assert tg2 is tg

    def test_assert_columns(self, node_stats_group):
        """ Tests for asserting a column or index exists """
        with pytest.raises(KeyError):
            node_stats_group.assert_column("notacolumn")
        node_stats_group.assert_column("phase_hint")

    def test_assert_columns_any_null(self, node_stats_group):
        """ Ensure a ValueError is raised if any nulls are found. """
        # setup
        df = node_stats_group.data.copy()
        df.loc[df.index[0], "station"] = None
        df["sampling_rate"] = np.NaN
        nsg = node_stats_group.new_from_dict(data=df)
        # this should not raise
        nsg.assert_column("station")
        # selecting any should cause it to raise
        with pytest.raises(ValueError):
            nsg.assert_column("station", raise_on_null="any")
        # but selecting all should not
        nsg.assert_column("station", raise_on_null="all")
        # if a column has all nulls it should raise on both
        with pytest.raises(ValueError):
            nsg.assert_column("sampling_rate", raise_on_null="any")
        with pytest.raises(ValueError):
            nsg.assert_column("sampling_rate", raise_on_null="all")
