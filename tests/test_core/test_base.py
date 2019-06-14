"""
Tests for functionality of parent Group classes.
"""
import pytest

import mopy


class TestNewFromDict:
    """ Tests for inplace keyword. """

    def test_new_trace_group(self, node_trace_group):
        """ ensure a new trace group is returned """
        tg = node_trace_group.new_from_dict()
        assert tg is not node_trace_group

    def test_inplace(self, node_trace_group):
        """ ensure inplace does not make a copy. """
        tg = node_trace_group.copy()
        tg2 = tg.new_from_dict(inplace=True)
        assert tg2 is tg
