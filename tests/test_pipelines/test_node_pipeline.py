"""
Tests for the local node Pipeline.
"""

import numpy as np
from numpy.testing import assert_allclose as np_assert
import pandas as pd
import pytest
from obsplus.utils.misc import register_func

from mopy.pipelines.node import LocalNodePipeLine

pipelines = []


@pytest.fixture(scope="session")
@register_func(pipelines)
def node_local_coal(node_inventory, node_st):
    """
    Init a node local pipeline on the coal node dataset.
    """
    pl = LocalNodePipeLine(
        inventory=node_inventory, waveforms=node_st, stream_processor=False,
    )
    return pl


@pytest.fixture(scope="session", params=pipelines)
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
        # check energy is additive
        energy = out["energy"].values
        p_s_energy = np.nansum(out[["energy_P", "energy_S"]], axis=1)
        assert np.allclose(energy, p_s_energy)
        # moments are median'ed
        moment = out["moment"].values
        p_s_moment = np.nanmedian(out[["moment_P", "moment_S"]], axis=1)
        np_assert(moment, p_s_moment)

    def test_create_catalog(self, node_local_coal, node_catalog):
        """The pipeline should be able to add magnitude objects to catalog."""
        mag_dict = node_local_coal.create_simple_magnitudes(node_catalog)
        assert isinstance(mag_dict, dict)
        assert len(mag_dict)

    def test_add_magnitudes_to_catalog(self, node_local_coal, node_catalog):
        """Tests for adding magnitudes to catalogs."""
        events = node_catalog.copy()
        out = node_local_coal.add_magnitudes(events, mw_preferred=True)
        assert len(out) == len(events)
        for ev1, ev2 in zip(node_catalog, out):
            assert len(ev1.magnitudes) < len(ev2.magnitudes)
