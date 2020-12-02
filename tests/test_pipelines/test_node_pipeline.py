"""
Tests for the local node Pipeline.
"""
from pathlib import Path

from functools import lru_cache
from typing import Union, Optional, List, Mapping, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import scipy
import obsplus
import obspy
import pandas as pd
import pytest
from obsplus.utils.misc import register_func

from mopy.pipelines.node import LocalNodePipeLine

pipelines = []

@pytest.fixture(scope='session')
@register_func(pipelines)
def node_local_crandall(crandall_bank, crandall_inventory):
    """
    Init a node local pipeline on crandall canyon.

    This violates some of the assumptions that the pipeline is for a local
    node network but allows testing the default response removal as well
    as fetching data from a bank rather than a stream.
    """
    pl = LocalNodePipeLine(
        inventory=crandall_inventory,
        waveforms=crandall_bank,
        stream_processor=False,
    )
    return pl

@pytest.fixture(scope='session')
@register_func(pipelines)
def node_local_coal(node_inventory, node_st):
    """
    Init a node local pipeline on the coal node dataset.
    """
    pl = LocalNodePipeLine(
        inventory=node_inventory,
        waveforms=node_st,
        stream_processor=False,
    )
    return pl


@pytest.fixture(scope='session', params=pipelines)
def pipeline(request) -> LocalNodePipeLine:
    """Meta fixture for collecting all pipelines"""
    return request.getfixturevalues(request.param)


class TestLocalNode:
    """Tests for the local node pipeline."""

    def test_calc_station_source_params(self, node_local_coal, node_catalog):
        """Calc source params per station/phase."""
        pipe = node_local_coal
        out = pipe.calc_station_source_parameters(node_catalog)
        assert isinstance(out, pd.DataFrame)
        assert len(out)

    def test_calc_source_params(self, node_local_coal, node_catalog):
        """Calc source params per event. """
        pipe = node_local_coal
        out = pipe.calc_source_parameters(node_catalog)
        breakpoint()


