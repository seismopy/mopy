"""
tests for the trace group
"""
from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose as np_assert
import pytest

from obspy import Stream
from obsplus.utils.time import to_utc

import mopy
import mopy.core.statsgroup
import mopy.core.tracegroup
from mopy import StatsGroup, TraceGroup, SpectrumGroup
from mopy.exceptions import NoPhaseInformationError
from mopy.testing import gauss


class TestBasics:

    @pytest.fixture(scope="function")
    def incomplete_trace(self, node_stats_group, node_st) -> Stream:
        """ Return a stream with part of the data missing for one of its traces """
        # Select a trace from a pick that is referenced in the stats group
        st = node_st.copy()
        pick = node_stats_group.data.iloc[0]
        seed_id = pick.name[-1]
        tr = st.select(id=seed_id)[0]
        # Trim the trace so it ends in the middle of the desired time window
        pick_start = pick["starttime"]
        pick_end = pick["endtime"]
        new_end = to_utc(pick_end) - (to_utc(pick_end) - to_utc(pick_start))/2
        tr.trim(tr.stats.starttime, new_end)  # Acts in place
        return st

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
        # create the stats group
        # TODO: if restrict_to_arrivals is True, this has different behavior
        #  when running the test separately vs as part of the entire test suite.
        #  This is probably something that should be investigated further, but
        #  is irrelevant to this specific test
        statsgroup = StatsGroup(node_catalog, node_inventory, restrict_to_arrivals=False)
        # A warning should be issued when it fails to find the station
        with pytest.warns(UserWarning):
            TraceGroup(statsgroup, st, motion_type="velocity")

    def test_incomplete_data_warns(self, node_stats_group, incomplete_trace):
        """
        Ensure an incomplete data stream warns and tosses the data
        """
        # A warning should be issued when it fails to find the station
        with pytest.warns(UserWarning):
            TraceGroup(node_stats_group, incomplete_trace, motion_type="velocity")

    def test_nan_time_windows(self, node_stats_group, node_st):
        """
        Make sure can gracefully handle having one or more events with missing time windows
        """
        stats_group = node_stats_group.copy()
        # Clear the time windows
        stats_group.data.starttime = np.nan
        stats_group.data.endtime = np.nan
        # Try to create a TraceGroup with the NaN time windows
        # For now this is going to fail, but I think it should maybe issue a warning instead?
        with pytest.raises(ValueError):
            TraceGroup(stats_group, node_st, motion_type="velocity")

    def test_empty_stats_group(self, node_stats_group_no_picks, node_st):
        """
        Make sure can gracefully handle a StatsGroup with no phase information
        """
        with pytest.raises(NoPhaseInformationError):
            TraceGroup(node_stats_group_no_picks, node_st, motion_type="velocity")


class TestDetrend:
    def _assert_zero_meaned(self, df):
        """ assert that the dataframe's columns have zero-mean. """
        mean = df.mean(axis=1)
        np.testing.assert_almost_equal(mean.values, 0)

    @pytest.fixture
    def tg_with_nan(self, node_trace_group):
        """ set some NaN value to node_trace_group. """
        df = node_trace_group.data.copy() + 1
        # make jagged NaN stuffs
        df.loc[:, df.columns[-4:]] = np.NaN
        df.loc[df.index[0], df.columns[-6] :] = np.NaN
        return node_trace_group.new_from_dict(data=df)

    """ Tests for removing the trend of data. """

    def test_constant_detrend(self, node_trace_group):
        out = node_trace_group.detrend(type="constant")
        self._assert_zero_meaned(out.data)

    def test_constant_with_nan(self, tg_with_nan):
        """ set some NaN values at end of trace and return. """
        out = tg_with_nan.detrend(type="constant")
        self._assert_zero_meaned(out.data)

    def test_linear_detrend(self, node_trace_group):
        """ make sure linear detrend works. """
        out = node_trace_group.detrend("linear")
        self._assert_zero_meaned(out.data)

    def test_linear_with_nan(self, tg_with_nan):
        """ ensure NaNs don't mess up linear detrend. """
        out = tg_with_nan.detrend("linear")
        self._assert_zero_meaned(out.data)


class TestToSpectrumGroup:
    """ Tests for converting the TraceGroup to SpectrumGroups. """

    @pytest.fixture
    def fft(self, node_trace_group):
        """ Convert the trace group to a spectrum group"""
        return node_trace_group.dft()

    @pytest.fixture
    def mtspec1(self, node_trace_group):
        """ Convert the trace group to a spectrum group via mtspec. """
        pytest.importorskip("mtspec")  # skip if mtspec is not installed
        return node_trace_group.mtspec()

    @pytest.fixture(params=("mtspec1", "fft"))
    def spec_from_trace(self, request):
        """ A gathering fixture for generic SpectrumGroup tests. """
        return request.getfixturevalue(request.param)

    # - General tests
    def test_type(self, spec_from_trace):
        """ Ensure the correct type was returned. """
        assert isinstance(spec_from_trace, SpectrumGroup)

    def test_compare_fft_mtspec(self, mtspec1, fft):
        """ fft and mtspec1 should not be radically different. """
        # calculate power sums
        df1, df2 = mtspec1.to_spectra_type("dft").data, abs(fft.data)
        sum1 = (df1 ** 2).sum()
        sum2 = (df2 ** 2).sum()
        # compare ratios between fft and mtspec
        ratio = sum1 / sum2
        assert abs(ratio.mean() - 1.0) < 0.1
        assert abs(ratio.median() - 1.0) < 0.1
