"""
Tests for channel information, specifically creating dataframes from it.
"""
from __future__ import annotations

from copy import deepcopy
from os.path import join

import numpy as np
import pandas as pd
import pytest
from obsplus.events.utils import get_seed_id
from obspy import UTCDateTime
from obspy.core.event import Pick, WaveformStreamID

import mopy
import mopy.core.statsgroup
from mopy.constants import CHAN_COLS
from mopy.testing import assert_not_nan


# --- Tests
class TestBasics:
    def test_type(self, node_channel_info):
        assert isinstance(node_channel_info, mopy.core.statsgroup.StatsGroup)

    def test_has_channels(self, node_channel_info):
        """ Test meta df """
        assert len(node_channel_info)

    def test_distance(self, node_channel_info):
        """ Test the channel info has reasonable distances. """
        df = node_channel_info.data
        dist = df["distance"]
        assert not dist.isnull().any()
        assert (dist > 0).all()

    def test_copy(self, node_channel_info):
        """ Ensure copying doesnt copy traces. """
        cop = node_channel_info.copy()
        # the base objects should have been copied
        assert id(cop) != id(node_channel_info)
        assert id(cop.data) != id(node_channel_info.data)

    def test_no_picks(self, node_channel_info_no_picks):
        assert len(node_channel_info_no_picks) == 0
        assert set(node_channel_info_no_picks.data.columns).issuperset(CHAN_COLS)

    def test_add_time_buffer(
        self, node_channel_info
    ):  # Not quite sure what's going on in this test...
        """
        Ensure time can be added to the start and end of the node_trace_group.
        """
        # Add times, start and end
        df = node_channel_info.data
        start = 1
        end = pd.Series(2, index=df.index)
        tg = node_channel_info.add_time_buffer(start=start, end=end)
        # Make sure a copy did occur
        assert tg is not node_channel_info
        # Make sure time offset is correct
        df2 = tg.data
        # Make sure to only get records with non-NaN start and end times
        df3 = df2.loc[df2["starttime"].notnull() & df2["endtime"].notnull()]
        df4 = df.loc[df3.index]
        assert ((df3["starttime"] + 1) == df4["starttime"]).all()
        assert ((df3["endtime"] - 2) == df4["endtime"]).all()

    def test_get_column_or_index(self, node_channel_info):
        """ tests for getting a series from a column or an index. """
        chan = node_channel_info.get_column_or_index("channel")
        assert isinstance(chan, pd.Series)
        assert chan.equals(node_channel_info.data["channel"])
        # first index
        name = node_channel_info.index.names[0]
        phase_hint = node_channel_info.get_column_or_index(name)
        assert isinstance(phase_hint, pd.Series)

    def test_assert_columns(self, node_channel_info):
        """ Tests for asserting a column or index exists """
        with pytest.raises(KeyError):
            node_channel_info.assert_has_column_or_index("notacolumn")
        node_channel_info.assert_has_column_or_index("phase_hint")


class TestSetPicks:
    """ Tests for the set_picks method """

    # Fixtures
    @pytest.fixture(scope="session")
    def pick_csv(self, data_path):  # Again, I would love a better way to do this...
        """ csv file containing pick information """
        return join(data_path, "picks.csv")

    @pytest.fixture(scope="session")
    def pick_df(self, pick_csv):
        """ single-indexed DataFrame containing pick information """
        return pd.read_csv(pick_csv)

    @pytest.fixture(scope="session")
    def bogus_pick_df(self, pick_csv):
        df = pd.read_csv(pick_csv)
        return df.rename(
            columns={"event_id": "random", "seed_id": "column", "phase": "name"}
        )

    @pytest.fixture(scope="session")
    def pick_df_multi(self, pick_csv):
        """ multi-indexed DataFrame containing pick information """
        return pd.read_csv(pick_csv, index_col=[0, 1, 2])

    @pytest.fixture(scope="session")
    def pick_df_extra(self, pick_csv):
        """ pick DataFrame with extra columns """
        df = pd.read_csv(pick_csv)
        df.loc[0, "extra"] = "abcd"
        return df

    @pytest.fixture(scope="session")
    def pick_dict(self, pick_df):
        """ mapping of pick meta-info to pick time """
        # Want to double check the column names here to make sure they match what ChannelInfo expects
        return {
            (pick.phase_hint, pick.event_id, pick.seed_id): pick.time
            for (_, pick) in pick_df.iterrows()
        }

    @pytest.fixture(scope="session")
    def picks(self, pick_dict):
        return {
            key: Pick(
                time=time,
                phase_hint=key[0],
                waveform_id=WaveformStreamID(seed_string=key[2]),
                onset="impulsive",
                polarity="negative",
            )
            for (key, time) in pick_dict.items()
        }

    @pytest.fixture(scope="session")
    def bad_event(self, node_inventory):
        net = node_inventory[0]
        sta = net[0]
        chan = sta[0]
        return {
            (
                "P",
                "not_an_event",
                f"{net.code}.{sta.code}.{chan.location_code}.{chan.code}",
            ): UTCDateTime(2019, 1, 1)
        }

    @pytest.fixture(scope="session")
    def bad_seed(self, node_catalog):
        return {
            ("P", node_catalog[0].resource_id.id, "bad seed"): UTCDateTime(2019, 1, 1)
        }

    @pytest.fixture(scope="function")
    def add_picks_from_dict(self, node_channel_info_no_picks, pick_dict):
        return node_channel_info_no_picks.set_picks(pick_dict)

    @pytest.fixture(scope="function")
    def add_picks_from_df(self, node_channel_info_no_picks, pick_df):
        return node_channel_info_no_picks.set_picks(pick_df)

    @pytest.fixture(scope="function")
    def add_picks_from_multi_df(self, node_channel_info_no_picks, pick_df_multi):
        return node_channel_info_no_picks.set_picks(pick_df_multi)

    @pytest.fixture(scope="function")
    def add_picks_from_extra_df(self, node_channel_info_no_picks, pick_df_extra):
        return node_channel_info_no_picks.set_picks(pick_df_extra)

    @pytest.fixture(scope="function")
    def add_picks_from_picks(self, node_channel_info_no_picks, picks):
        return node_channel_info_no_picks.set_picks(picks)

    # Tests
    def test_set_picks(self, add_picks_from_dict, pick_dict):
        """ verify that it is possible to attach picks from a mapping (dictionary) """
        # Make sure new picks actually got attached
        assert len(add_picks_from_dict) == len(pick_dict)
        # Make sure the attached data are what you expect
        newest = add_picks_from_dict.data.iloc[-1]
        pick = pick_dict[newest.name]
        seed_id = newest.name[2].split(".")
        expected_dict = {
            "network": seed_id[0],
            "station": seed_id[1],
            "location": seed_id[2],
            "channel": seed_id[3],
            "time": pick,
            "phase_hint": newest.name[0],
        }
        numbers = {
            "distance": 384.21949334560514,
            "azimuth": 138.9411665659937,
            "horizontal_distance": 133.39459009552598,
            "depth_distance": 360.32000000000016,
            "ray_path_length": 384.21949334560514,
            "velocity": 1800.0,
            "radiation_coefficient": 0.59999999999999998,
            "quality_factor": 400,
            "spreading_coefficient": 384.21949334560514,
            "density": 3000,
            "shear_modulus": 2200000000,
            "free_surface_coefficient": 2.0,
        }
        nans = ["method_id", "endtime", "starttime", "sampling_rate", "onset", "polarity"]
        for item, value in expected_dict.items():
            assert newest[item] == value
        for item, value in numbers.items():
            assert np.isclose(newest[item], value)
        for item in nans:
            assert np.isnan(newest[item])
        assert_not_nan(newest["pick_id"])

    def test_bogus_picks(self, node_channel_info_no_picks):
        """ verify fails predictably if something other than an allowed pick format gets passed """
        with pytest.raises(TypeError):
            node_channel_info_no_picks.set_picks(2)

    def test_bad_pick_dict(self, node_channel_info_no_picks):
        """ verify fails predictably if a malformed pick dictionary gets provided """
        # Bogus key
        pick_dict = {"a": UTCDateTime("2018-09-12")}
        with pytest.raises(TypeError):
            node_channel_info_no_picks.set_picks(pick_dict)

        # Bogus value
        pick_dict = {
            (
                "P",
                "event_1",
                node_channel_info_no_picks.event_station_info.index.levels[1][0],
            ): "a"
        }
        with pytest.raises(TypeError):
            node_channel_info_no_picks.set_picks(pick_dict)

    def test_set_picks_existing_picks_overwrite(self, add_picks_from_dict, pick_dict):
        """ verify the process of overwriting picks on ChannelInfo """
        # Make sure to issue a warning to the user that it is overwriting
        pick_dict = deepcopy(pick_dict)
        for key in pick_dict:
            pick_dict[key] = UTCDateTime(pick_dict[key]) + 1
        with pytest.warns(UserWarning):
            out = add_picks_from_dict.set_picks(pick_dict)
        assert len(out) == len(pick_dict)
        for index in pick_dict:
            assert out.data.loc[index, "time"] == pick_dict[index]

    def test_set_picks_dataframe(self, add_picks_from_df, pick_df):
        """ verify that it is possible to attach picks from a DataFrame/csv file """
        assert len(add_picks_from_df) == len(pick_df)
        # Make sure the resulting data is what you expect
        newest = add_picks_from_df.data.iloc[-1]
        pick_time = (
            pick_df.set_index(["phase_hint", "event_id", "seed_id"])
            .loc[newest.name]
            .time
        )
        assert newest.time == UTCDateTime(pick_time)
        assert_not_nan(newest.pick_id)

    def test_multi_indexed_dataframe(self, add_picks_from_multi_df, pick_df_multi):
        """ verify that it is possible to use a multi-indexed dataframe with the pick information """
        # Things may be starting to get more complicated than I actually want to deal with here
        assert len(add_picks_from_multi_df) == len(pick_df_multi)
        # Make sure the input data is what you expect
        newest = add_picks_from_multi_df.data.iloc[-1]
        pick_time = (
            pick_df_multi.reset_index()
            .set_index(["phase_hint", "event_id", "seed_id"])
            .loc[newest.name]
            .time
        )
        assert newest.time == UTCDateTime(pick_time)
        assert_not_nan(newest.pick_id)

    def test_picks_df_extra_cols(self, add_picks_from_extra_df, pick_df_extra):
        add_picks_from_extra_df.set_picks(pick_df_extra)
        assert "extra" in add_picks_from_extra_df.data.columns

    def test_invalid_df(self, node_channel_info_no_picks, bogus_pick_df):
        """ verify fails predictably if a df doesn't contain the required info """
        with pytest.raises(KeyError):
            node_channel_info_no_picks.set_picks(bogus_pick_df)

    def test_df_overwrite(self, add_picks_from_df, pick_df):
        """ verify that my hack for preserving pick_ids when overwriting picks with a df works as expected """
        pick_df = deepcopy(pick_df)
        resource_ids = add_picks_from_df.data.pick_id
        for num, row in pick_df.iterrows():
            pick_df.loc[num, "time"] = UTCDateTime(row.time) + 1
        with pytest.warns(UserWarning):
            out = add_picks_from_df.set_picks(pick_df)
        # Make sure the existing picks were overwritten, not appended
        assert len(out) == len(pick_df)
        # Make sure the times got updated
        for num, row in pick_df.iterrows():
            assert (
                out.data.loc[(row.phase_hint, row.event_id, row.seed_id), "time"]
                == row.time
            )
        # Make sure the resource_ids haven't changed
        assert (add_picks_from_df.data.pick_id == resource_ids).all()

    def test_set_picks_from_picks(self, add_picks_from_picks, picks):
        """ verify that it is possible to attach picks from a mapping of pick objects """
        # This will likely require some refactoring after Derrick pushes his changes
        assert len(add_picks_from_picks) == len(picks)
        # Make sure the input data is what you expect
        newest = add_picks_from_picks.data.iloc[-1]
        pick = picks[newest.name]
        seed_id = get_seed_id(pick).split(".")
        expected_dict = {
            "network": seed_id[0],
            "station": seed_id[1],
            "location": seed_id[2],
            "channel": seed_id[3],
            "time": pick.time,
            "onset": pick.onset,
            "polarity": pick.polarity,
            "pick_id": pick.resource_id.id,
            "phase_hint": pick.phase_hint,
        }
        numbers = {
            "distance": 384.21949334560514,
            "azimuth": 138.9411665659937,
            "horizontal_distance": 133.39459009552598,
            "depth_distance": 360.32000000000016,
            "ray_path_length": 384.21949334560514,
            "velocity": 1800.0,
            "radiation_coefficient": 0.59999999999999998,
            "quality_factor": 400,
            "spreading_coefficient": 384.21949334560514,
            "density": 3000,
            "shear_modulus": 2200000000,
            "free_surface_coefficient": 2.0,
        }
        nans = ["method_id", "endtime", "starttime", "sampling_rate"]
        for item, value in expected_dict.items():
            assert newest[item] == value
        for item, value in numbers.items():
            assert np.isclose(newest[item], value)
        for item in nans:
            assert np.isnan(newest[item])

    def test_copied(self, add_picks_from_dict, node_channel_info_no_picks):
        """ verify that set_picks returns a copy of itself by default """
        assert not id(add_picks_from_dict) == id(node_channel_info_no_picks)
        assert not id(add_picks_from_dict.data) == id(node_channel_info_no_picks.data)

    def test_inplace(self, node_channel_info_no_picks, pick_dict):
        length = len(node_channel_info_no_picks)
        node_channel_info_no_picks.set_picks(pick_dict, inplace=True)
        assert len(node_channel_info_no_picks) > length

    # TODO: For now it's on the user to convert it to a DataFrame
    #    def test_set_picks_phase_file(self, node_channel_info_no_picks):
    #        """ verify it is possible to attach picks using a hypoinverse phase file or other format? """
    #        assert False

    def test_event_doesnt_exist(self, node_channel_info_no_picks, bad_event):
        """ verify that it fails predictably if the specified event does not exist"""
        with pytest.raises(KeyError):
            node_channel_info_no_picks.set_picks(bad_event)

    def test_seed_doesnt_exist(self, node_channel_info_no_picks, bad_seed):
        with pytest.raises(KeyError):
            node_channel_info_no_picks.set_picks(bad_seed)

    def test_instant_gratification(self):
        assert True


# TODO: This is kinda important
# class TestSetTimeWindows:
#    """ Yeah... """

# class TestSetVelocity:
#     """ Tests for the set_velocity method """
#
#     # Fixtures
#
#     # Tests
#     def test_set_velocity(self, node_channel_info):
#         """ verify that it is possible to specify an arbitrary phase velocity """
#         assert False
#
#     # TODO: this is very much a stretch goal
# #    def test_set_velocity_from_3d(self, node_channel_info):
# #        """ verify that it is possible to get a velocity from a spatially varying model (using obsplus.Grid) """
# #        assert False
#
#     def test_set_velocity_from_config(self, node_channel_info):  # I'm not sure if we want to have these here or elsewhere...
#         """ verify that it is possible to retrieve a velocity from a callable in .mopy.py """
#         assert False
#
#
# class TestSetDensity:
#     """ Tests for the set_density method """
#
#     # Fixtures
#
#     # Tests
#     def test_set_density(self, node_channel_info):
#         assert False
#
#     # TODO: this is very much a stretch goal
#     # def test_set_density_from_3d(self, node_channel_info):
#     #     """ verify that it is possible to get a density from a spatially varying model (using obsplus.Grid """
#     #     assert False
#
#     def test_set_density_from_config(self, node_channel_info):
#         assert False
#
#
# class TestSetGeometricSpreading:
#     """ Tests for the set_geometric_spreading method """
#
#     # Fixtures
#
#     # Tests
#     def test_set_geometric_spreading_predefined(self, node_channel_info):
#         assert False
#
#     def test_set_geometric_spreading_from_callable(self, node_channel_info):
#         assert False
#
#     def test_set_geometric_spreading_from_config(self, node_channel_info):
#         assert False
#
#
# class TestSetAttenuation:
#     """ Tests for the set_attenuation method """
#
#     # Fixtures
#
#     # Tests
#     def test_set_attenuation(self, node_channel_info): # Do we want this to be phase-specific???
#         assert False
#
#     def test_set_attenuation_from_callable(self, node_channel_info):
#         assert False
#
#     def test_set_attenuation_from_config(self, node_channel_info):
#         assert False
