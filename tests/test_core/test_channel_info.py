"""
Tests for channel information, specifically creating dataframes from it.
"""
import pandas as pd
import pytest
from os.path import join

from obspy import UTCDateTime

import mopy
import mopy.core.channelinfo
from mopy.constants import CHAN_COLS


# --- Tests
class TestBasics:
    def test_type(self, node_channel_info):
        assert isinstance(node_channel_info, mopy.core.channelinfo.ChannelInfo)

    def test_has_traces(self, node_channel_info):
        """ Test meta df """
        data_df = node_channel_info.data
        assert set(CHAN_COLS).issubset(data_df.columns)

    def test_distance(self, node_channel_info):
        """ Test the channel info has reasonable distances. """
        df = node_channel_info.data
        dist = df['distance']
        assert not dist.isnull().any()
        assert (dist > 0).all()

    def test_copy(self, node_channel_info):
        """ Ensure copying doesnt copy traces. """
        cop = node_channel_info.copy()
        # the base objects should have been copied
        assert id(cop) != id(node_channel_info)
        assert id(cop.data) != id(node_channel_info.data)
        # make sure the values are equal
        df1 = node_channel_info.data
        df2 = cop.data
        assert df1.drop(columns='trace').equals(df2.drop(columns='trace'))
        # make sure traces weren't copied
        tr_id1 = {id(tr) for tr in df1['trace']}
        tr_id2 = {id(tr) for tr in df2['trace']}
        assert tr_id1 == tr_id2


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
    def pick_df_multi(self, pick_csv):
        """ multi-indexed DataFrame containing pick information """
        return pd.read_csv(pick_csv, index_col=[0, 1, 2])

    @pytest.fixture(scope="session")
    def pick_dict(self, pick_df):
        """ mapping of pick meta-info to pick time """
        # Want to double check the column names here to make sure they match what ChannelInfo expects
        return {(pick.phase, pick.event_id, pick.seed_id): pick.time for _, pick in pick_df.iterrows()}

    @pytest.fixture(scope="session")
    def bad_event(self, node_inventory):
        net = node_inventory[0]
        sta = net[0]
        chan = sta[0]
        return {("P", "not_an_event", f"{net.code}.{sta.code}.{chan.location_code}.{chan.code}"):
                UTCDateTime(2019, 1, 1)}

    @pytest.fixture(scope="session")
    def bad_seed(self, node_catalog):
        return {("P", node_catalog[0].resource_id.id, "bad seed"): UTCDateTime(2019, 1, 1)}

    @pytest.fixture(scope="function")
    def node_channel_info_no_picks(self, node_catalog_no_picks, node_inventory):
        """ return a ChannelInfo for a catalog that doesn't have any picks """
        # This will probably need to be moved and/or refactored in the future, but for now...
        kwargs = dict(catalog=node_catalog_no_picks, inventory=node_inventory)
        return mopy.core.channelinfo.ChannelInfo(**kwargs)

    # Tests
    def test_set_picks(self, node_channel_info_no_picks, pick_dict):
        """ verify that it is possible to attach picks from a mapping (dictionary) """
        node_channel_info_no_picks.set_picks(pick_dict)
        assert len(node_channel_info_no_picks.data) == len(pick_dict)

    def test_bogus_picks(self, node_channel_info_no_picks):
        """ verify fails predictably if something other than an allowed pick format gets passed """
        with pytest.raises(TypeError):
            node_channel_info_no_picks.set_picks(2)

    def test_bad_pick_dict(self, node_channel_info_no_picks):
        """ verify fails predictably if a malformed pick dictionary gets provided """
        # Bogus key
        pick_dict = {"a", UTCDateTime("2018-09-12")}
        with pytest.raises(TypeError):
            node_channel_info_no_picks.set_picks(pick_dict)

        # Bogus value
        pick_dict = {("sensible info", "goes", "here"): "a"}
        with pytest.raises(TypeError):
            node_channel_info_no_picks.set_picks(pick_dict)

    def test_set_picks_existing_picks_overwrite(self, node_channel_info_no_picks, pick_dict):
        """ verify the process of overwriting picks on ChannelInfo """
        # Make sure to issue a warning to the user that it is overwriting
        node_channel_info_no_picks.set_picks(pick_dict)
        with pytest.warns(UserWarning):
            node_channel_info_no_picks.set_picks(pick_dict)
        assert len(node_channel_info_no_picks.data) == len(pick_dict)

    def test_set_picks_existing_picks_append(self, node_channel_info, pick_dict):
        """ verify the process of appending picks to a ChannelInfo that already has picks """
        initial_len = len(node_channel_info)
        node_channel_info.set_picks(pick_dict)
        assert len(node_channel_info.data) == len(pick_dict) + initial_len

    def test_set_picks_dataframe(self, node_channel_info_no_picks, pick_df):
        """ verify that it is possible to attach picks from a DataFrame/csv file """
        node_channel_info_no_picks.set_picks(pick_df)
        assert len(node_channel_info_no_picks.data) == len(pick_df)

    def test_multi_indexed_dataframe(self, node_channel_info_no_picks, pick_df_multi):
        """ verify that it is possible to use a multi-indexed dataframe with the pick information """
        # Things may be starting to get more complicated than I actually want to deal with here
        node_channel_info_no_picks.set_picks(pick_df_multi)
        assert len(node_channel_info_no_picks.data) == len(pick_df_multi)

    def test_invalid_df(self, node_channel_info_no_picks, bogus_pick_df):
        """ verify fails predictably if a df doesn't contain the required info """
        with pytest.raises(KeyError):
            node_channel_info_no_picks.set_picks(bogus_pick_df)

    def test_set_picks_from_picks(self, node_channel_info_no_picks, picks):
        """ verify that it is possible to attach picks from a mapping of pick objects """
        node_channel_info_no_picks.set_picks(picks)
        assert len(node_channel_info_no_picks.data) == len(picks)

    # TODO: For now it's on the user to convert it to a DataFrame
    #    def test_set_picks_phase_file(self, node_channel_info_no_picks):
    #        """ verify it is possible to attach picks using a hypoinverse phase file or other format? """
    #        assert False

    def test_event_doesnt_exist(self, node_channel_info_no_picks, bad_event):
        """ verify that it fails predictably if the specified event does not exist"""

        with pytest.raises(KeyError):
            node_channel_info_no_picks.set_picks(bad_event)

    def test_seed_doesnt_exist(self, node_channel_info_no_picks, bad_seed):
        bad_seed = {("P", "yikes...", "bad_seed"): UTCDateTime(2019, 1, 1)}
        with pytest.raises(KeyError):
            node_channel_info_no_picks.set_picks(bad_seed)

# TODO: This is kinda important
#class TestSetTimeWindows:
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
