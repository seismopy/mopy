"""
Tests for miscellaneous utility functions.
"""
import numpy as np
import pandas as pd
import pytest

import mopy
import mopy.utils.misc as misutil


class TestOptionalImport:
    """ Tests for optional module imports. """

    def test_good_import(self):
        """ Test importing a module that should work. """
        mod = misutil.optional_import("mopy")
        assert mod is mopy

    def test_bad_import(self):
        """ Test importing a module that does not exist """
        with pytest.raises(ImportError, match="is not installed"):
            misutil.optional_import("areallylongbadmodulename")


class TestPadOrTrim:
    """ Test that padding or trimming a numpy array. """

    def test_trim(self):
        """
        Ensure an array gets trimmed when sample count is smaller than last
        dimension.
        """
        ar = np.random.rand(10, 10)
        out = misutil.pad_or_trim(ar, 1)
        assert np.shape(out)[-1] == 1

    def test_fill_zeros(self):
        """ Tests for filling array with zeros. """
        ar = np.random.rand(10, 10)
        in_dtype = ar.dtype
        out = misutil.pad_or_trim(ar, sample_count=15)
        assert np.shape(out)[-1] == 15
        assert out.dtype == in_dtype, "datatype should not change"
        assert np.all(out[:, 10:] == 0)

    def test_fill_nonzero(self):
        """ Tests for filling non-zero values """
        ar = np.random.rand(10, 10)
        out = misutil.pad_or_trim(ar, sample_count=15, pad_value=np.NaN)
        assert np.all(np.isnan(out[:, 10:]))


class TestBroadcastParameter:
    """
    Tests for a function to broadcast a parameter over a column of a
    multiindexed DataFrame
    """

    velocity = 8000
    velocity_dict = {"P": 5000, "S": 3000}

    # Fixtures
    @pytest.fixture(scope="function")
    def broadcast_df(self, node_stats_group) -> pd.DataFrame:
        """ Return a dataframe to use for testing"""
        return node_stats_group.data.copy()

    @pytest.fixture(scope="function")
    def mapped_source_velocities(self, node_stats_group) -> pd.Series:
        """ Returns a series of source velocities to map to the StatsGroup"""
        ind = node_stats_group.index
        return pd.Series(data=[x * 100 for x in range(len(ind))], index=ind)

    @pytest.fixture(scope="function")
    def simple_velocity(self, broadcast_df) -> pd.DataFrame:
        """ Returns a DataFrame with a uniform velocity applied"""
        return misutil.broadcast_param(
            broadcast_df, self.velocity, "source_velocity", "phase_hint"
        )

    # Tests
    def test_single_value(self, simple_velocity):
        """ verify that it is possible to specify a float velocity """
        assert (simple_velocity["source_velocity"] == self.velocity).all()

    def test_broadcast_from_dict(self, broadcast_df):
        """ verify that it is possible to specify source velocities from a dict"""
        out = misutil.broadcast_param(
            broadcast_df, self.velocity_dict, "source_velocity", "phase_hint"
        )
        for phase, vel in self.velocity_dict.items():
            assert (out.xs(phase, level="phase_hint")["source_velocity"] == vel).all()

    def test_broadcast_from_dict_not_possible(self, broadcast_df):
        """Ensure broadcasting from dict raises."""
        with pytest.raises(TypeError, match="not supported"):
            misutil.broadcast_param(
                broadcast_df, self.velocity_dict, "source_velocity", None
            )

    def test_broadcast_from_series(self, broadcast_df, mapped_source_velocities):
        """Ensure broadcasting from series works."""
        out = misutil.broadcast_param(
            broadcast_df, mapped_source_velocities, "source_velocity", "phase_hint"
        )
        assert out["source_velocity"].equals(mapped_source_velocities)

    def test_set_velocity_bogus(self, broadcast_df):
        """ verify that a bogus velocity fails predictably"""
        with pytest.raises(TypeError):
            misutil.broadcast_param(
                broadcast_df, "bogus", "source_velocity", "phase_hint"
            )

    def test_set_velocity_no_picks(self):
        """
        make sure it is not possible to set velocities if no picks have been
        provided
        """
        with pytest.raises(ValueError, match="No phases have been added"):
            misutil.broadcast_param(
                pd.DataFrame(), self.velocity_dict, "source_velocity", "phase_hint"
            )

    def test_set_velocity_overwrite(self, simple_velocity):
        """ make sure overwriting issues a warning """
        with pytest.warns(UserWarning, match="Overwriting"):
            out = misutil.broadcast_param(
                simple_velocity,
                self.velocity_dict,
                "source_velocity",
                "phase_hint",
                na_only=False,
            )
        assert not (out["source_velocity"] == self.velocity).any()
