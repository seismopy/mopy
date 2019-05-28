"""
Tests for utility functions.
"""
import numpy as np
import obsplus
import obspy
import pandas as pd
import pytest

import mopy
import mopy.utils as utils
from mopy.constants import MOTION_TYPES


class TestTraceToDF:
    """ Tests for converting traces to dataframes. """

    @pytest.fixture
    def vel_trace(self):
        """ Return the first example trace with response removed. """
        tr = obspy.read()[0]
        inv = obspy.read_inventory()
        tr.remove_response(inventory=inv, output="VEL")
        return tr

    @pytest.fixture
    def df(self, vel_trace):
        """ return a dataframe from the example trace. """
        return utils.trace_to_spectrum_df(vel_trace, "velocity")

    def test_type(self, df):
        """ ensure a dataframe was returned. """
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(MOTION_TYPES)
        assert not df.empty

    def test_freq_count_shorten(self, vel_trace):
        """ ensure the frequency count returns a df with the correct
        number of frequencies when shortened. """
        df = utils.trace_to_spectrum_df(vel_trace, "velocity", freq_count=100)
        assert len(df.index) == 101

    def test_freq_count_lengthen(self, vel_trace):
        """ ensure the zero padding takes place to lengthen dfs. """
        tr_len = len(vel_trace.data)
        df = utils.trace_to_spectrum_df(vel_trace, "velocity", freq_count=tr_len + 100)
        assert len(df.index) == tr_len + 101


class TestPickandDurations:
    """ tests for extracting picks and durations from events. """

    @pytest.fixture
    def pick_duration_df(self, crandall_event):
        """ return the pick_durations stream from crandall. """
        return utils.get_phase_window_df(
            crandall_event, min_duration=0.2, max_duration=2
        )

    def test_basic(self, pick_duration_df, crandall_event):
        """ Make sure correct type was returned and df has expected len. """
        df = pick_duration_df
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_dict(self, crandall_event, crandall_stream):
        """ test that min_duration can be a dictionary. """
        st = crandall_stream
        # ensure at least 40 samples are used
        min_dur = {tr.id: 40 / tr.stats.sampling_rate for tr in st}
        df = utils.get_phase_window_df(crandall_event, min_duration=min_dur)
        assert not df.tw_start.isnull().any()
        assert not df.tw_end.isnull().any()

    def test_all_channels_included(self, node_dataset):
        """ ensure all the channels of the same instrument are included. """
        # get a pick dataframe
        event = node_dataset.event_client.get_events()[0]
        # now get a master stream
        time = obsplus.get_reference_time(event)
        t1, t2 = time - 1, time + 15
        st = node_dataset.waveform_client.get_waveforms(starttime=t1, endtime=t2)
        id_sequence = {tr.id for tr in st}
        #
        out = utils.get_phase_window_df(event=event, channel_codes=id_sequence)
        # iterate the time and ensure each has all channels
        for time, df in out.groupby("time"):
            assert len(df) == 3
            assert len(df["seed_id"]) == len(set(df["seed_id"])) == 3
        # make sure no stuff is duplicated
        assert not out.duplicated(["phase_hint", "seed_id"]).any()


class TestOptionalImport:
    """ Tests for optional module imports. """

    def test_good_import(self):
        """ Test importing a module that should work. """
        mod = utils.optional_import("mopy")
        assert mod is mopy

    def test_bad_import(self):
        """ Test importing a module that does not exist """
        with pytest.raises(ImportError) as e:
            _ = utils.optional_import("areallylongbadmodulename")
        msg = str(e.value)
        assert "is not installed" in msg


class TestPadOrTrim:
    """ Test that padding or trimming a numpy array. """

    def test_trim(self):
        """
        Ensure an array gets trimmed when sample count is smaller than last
        dimension.
        """
        ar = np.random.rand(10, 10)
        out = utils.pad_or_trim(ar, 1)
        assert np.shape(out)[-1] == 1

    def test_fill_zeros(self):
        """ Tests for filling array with zeros. """
        ar = np.random.rand(10, 10)
        in_dtype = ar.dtype
        out = utils.pad_or_trim(ar, sample_count=15)
        assert np.shape(out)[-1] == 15
        assert out.dtype == in_dtype, "datatype should not change"
        assert np.all(out[:, 10:] == 0)

    def test_fill_nonzero(self):
        """ Tests for filling non-zero values """
        ar = np.random.rand(10, 10)
        out = utils.pad_or_trim(ar, sample_count=15, pad_value=np.NaN)
        assert np.all(np.isnan(out[:, 10:]))
