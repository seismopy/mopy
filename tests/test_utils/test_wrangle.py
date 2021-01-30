#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for data-wrangling utilities
"""
from copy import deepcopy
from typing import Tuple

import obsplus
import obspy
from obspy.core.event import Event, ResourceIdentifier
import pandas as pd
import pytest

import mopy
from mopy.constants import MOTION_TYPES
import mopy.utils.wrangle


class TestTraceToDF:
    """ Tests for converting traces to dataframes. """

    @pytest.fixture
    def vel_trace(self) -> obspy.Trace:
        """ Return the first example trace with response removed. """
        tr = obspy.read()[0]
        inv = obspy.read_inventory()
        tr.remove_response(inventory=inv, output="VEL")
        return tr

    @pytest.fixture
    def df(self, vel_trace) -> pd.DataFrame:
        """ return a dataframe from the example trace. """
        return mopy.utils.wrangle.trace_to_spectrum_df(vel_trace, "velocity")

    def test_type(self, df):
        """ ensure a dataframe was returned. """
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(MOTION_TYPES)
        assert not df.empty

    def test_freq_count_shorten(self, vel_trace):
        """
        Ensure the frequency count returns a df with the correct
        number of frequencies when shortened.
        """
        df = mopy.utils.wrangle.trace_to_spectrum_df(
            vel_trace, "velocity", freq_count=100
        )
        assert len(df.index) == 101

    def test_freq_count_lengthen(self, vel_trace):
        """ ensure the zero padding takes place to lengthen dfs. """
        tr_len = len(vel_trace.data)
        df = mopy.utils.wrangle.trace_to_spectrum_df(
            vel_trace, "velocity", freq_count=tr_len + 100
        )
        assert len(df.index) == tr_len + 101


class TestPickandDurations:
    """ tests for extracting picks and durations from events. """

    @pytest.fixture
    def crandall_event_eval_status(self, crandall_event) -> Event:
        """
        Sets the evaluation_status to confirmed for every other pick
        associated with the arrivals
        """
        eve = crandall_event.copy()
        # It's necessary to do it this way to avoid ResourceIdentifier problems...
        arrs = eve.arrivals_to_df().iloc[::2]["pick_id"].values
        for p in eve.picks:
            if p.resource_id.id in arrs:
                p.evaluation_status = "confirmed"
        return eve

    @pytest.fixture
    def crandall_s_before_p(self, crandall_event) -> Tuple[Event, int]:
        """
        Set the S-pick time to be before the P-pick time for a couple of stations
        Parameters
        """
        # Make a copy of the event
        eve = crandall_event.copy()
        # Fix the phase hints to simply be P or S and set the evaluation status to reviewed
        for pick in eve.picks:
            if pick.phase_hint in {"P", "Pb"}:
                pick.phase_hint = "P"
            elif pick.phase_hint in {"S", "Sb"}:
                pick.phase_hint = "S"
            # pick.evaluation_status = "reviewed"
        # Get a list of picks for stations that have both P- and S-picks
        picks = eve.picks_to_df()
        reviewed = picks.loc[(picks["evaluation_status"] == "reviewed")]
        picks_to_modify = reviewed.loc[~(picks["phase_hint"] == "?")]
        picks_to_modify = picks_to_modify.groupby("station").filter(
            lambda x: len(x) == 2
        )
        picks_to_modify = picks_to_modify.loc[
            picks_to_modify["phase_hint"] == "S"
        ].resource_id
        # Move the S-picks before the P-picks
        for num, p in picks_to_modify.items():
            pick = ResourceIdentifier(p).get_referred_object()
            pick.time -= 120
        number_picks = len(reviewed) - len(picks_to_modify) * 2
        return eve, number_picks

    @pytest.fixture
    def pick_duration_df(self, crandall_event_eval_status):
        """ return the pick_durations stream from crandall. """
        return mopy.utils.wrangle.get_phase_window_df(
            crandall_event_eval_status, min_duration=0.2, max_duration=2
        )

    def test_basic(self, pick_duration_df, crandall_event_eval_status):
        """ Make sure correct type was returned and df has expected len. """
        df = pick_duration_df
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 17

    def test_dict(self, crandall_event_eval_status, crandall_stream):
        """ test that min_duration can be a dictionary. """
        st = crandall_stream
        # ensure at least 40 samples are used
        min_dur = {tr.id: 40 / tr.stats.sampling_rate for tr in st}
        df = mopy.utils.wrangle.get_phase_window_df(
            crandall_event_eval_status, min_duration=min_dur
        )
        assert len(df)
        assert not df.starttime.isnull().any()
        assert not df.endtime.isnull().any()

    def test_all_channels_included(self, node_dataset):
        """ ensure all the channels of the same instrument are included. """
        # get a pick dataframe
        event = node_dataset.event_client.get_events()[0]
        # now get a master stream
        time = obsplus.get_reference_time(event)
        t1, t2 = time - 1, time + 15
        st = node_dataset.waveform_client.get_waveforms(starttime=t1, endtime=t2)
        # Mock up a set of channel codes
        id_sequence = {(tr.id[:-1], tr.id) for tr in st}
        #
        out = mopy.utils.wrangle.get_phase_window_df(
            event=event, channel_codes=id_sequence, restrict_to_arrivals=False
        )
        # iterate the time and ensure each has all channels
        for time, df in out.groupby("time"):
            assert len(df) == 3
            assert len(df["seed_id"]) == len(set(df["seed_id"])) == 3
        # make sure no stuff is duplicated
        assert not out.duplicated(["phase_hint", "seed_id"]).any()

    def test_s_before_p(self, crandall_s_before_p):
        """ Make sure that if an S-pick is before a P-pick, neither gets used """
        eve = crandall_s_before_p[0]
        num_picks = crandall_s_before_p[1]

        with pytest.warns(UserWarning, match="S-pick is earlier than P-pick"):
            df = mopy.utils.wrangle.get_phase_window_df(
                eve, min_duration=0.2, max_duration=2, restrict_to_arrivals=False
            )
        assert len(df.loc[df["phase_hint"] != "Noise"]) == num_picks
