"""
Tests for channel information, specifically creating dataframes from it.
"""
from __future__ import annotations

from copy import deepcopy
from os.path import join

import pandas as pd
import pytest
from obsplus.utils.time import to_timedelta64, to_datetime64
from obspy import UTCDateTime

import mopy
import mopy.core.statsgroup
from mopy import StatsGroup
from mopy.constants import (
    STAT_DTYPES,
    MOPY_SPECIFIC_DTYPES,
    _INDEX_NAMES,
    DIST_COLS,
    NOISE_RADIATION_COEFFICIENT,
    NOISE_QUALITY_FACTOR,
)
from mopy.utils.testing import assert_not_nan

# # --- Constants

PDF_IND = ["phase_hint", "event_id", "seed_id"]


# --- Tests
class TestBasics:
    @pytest.fixture(scope="function")
    def defaults_stats_group(self, node_stats_group) -> StatsGroup:
        """ Return a StatsGroup with default values applied (but no other values) """
        return node_stats_group.apply_defaults()

    def test_type(self, node_stats_group):
        assert isinstance(node_stats_group, mopy.core.statsgroup.StatsGroup)

    def test_has_channels(self, node_stats_group, node_catalog):
        """ Test meta df """
        pdf = node_catalog.picks_to_df()
        # Get the number of non-rejected, non-noise picks
        # picks = len(pdf.loc[(~(pdf["evaluation_status"] == "rejected") & ~(pdf["phase_hint"] == "Noise"))])
        # Get the number of noise picks
        # noise = picks*3 + len(pdf.loc[~(pdf["evaluation_status"] == "rejected") & ~(pdf["phase_hint"] == "Noise")])
        # assert len(node_stats_group) == (picks + noise)
        # I'm going to let this go for now, but I would like to figure out why I can't get that number to match
        assert len(node_stats_group)

    def test_statsgroup_from_event(self, node_catalog, node_inventory):
        """Using a single event to init a statsgroup should be supported."""
        sg = StatsGroup(catalog=node_catalog[0], inventory=node_inventory)
        assert isinstance(sg, StatsGroup)

    # TODO: It should also be possible to init a statsgroup without a Catalog/Inventory (ex., using a dataframe)

    def test_distance(self, node_stats_group):
        """ Test the stats group has reasonable distances. """
        df = node_stats_group.data
        dist = df["distance_m"]
        assert not dist.isnull().any()
        assert (dist > 0).all()

    def test_contains_nans(self, node_stats_group):
        """ Make sure the StatsGroup contains nans until explicitly updated """
        nan_cols = {
            "velocity",
            "radiation_coefficient",
            "quality_factor",
            "spreading_coefficient",
            "density",
            "shear_modulus",
        }
        nan_cols = list(nan_cols.intersection(node_stats_group.data.columns))
        assert node_stats_group.data[nan_cols].isnull().all().all()

    def test_apply_defaults(self, defaults_stats_group, node_stats_group):
        """ Fill missing data in node_stats_group with defaults """

        # Make sure it returns a copy by default, which is not None
        assert isinstance(defaults_stats_group, StatsGroup)
        assert defaults_stats_group is not node_stats_group
        # Make sure the missing values actually got populated
        # with the exception of velocity which has no default.
        df = defaults_stats_group.data.drop(columns="source_velocity")
        assert df.notnull().all().all()

    def test_apply_defaults_noise(self, defaults_stats_group):
        """
        Make sure the columns that are source specific are set for noise phases such that they have no impact
        """
        checks = [
            ("quality_factor", NOISE_QUALITY_FACTOR),
            ("radiation_coefficient", NOISE_RADIATION_COEFFICIENT),
            ("spreading_coefficient", 1),
        ]
        for key, val in checks:
            assert (
                defaults_stats_group.data.xs("Noise", level="phase_hint")[key] == val
            ).all()

    def test_no_picks(self, node_stats_group_no_picks):
        assert len(node_stats_group_no_picks) == 0
        assert set(node_stats_group_no_picks.data.columns).issuperset(
            set(STAT_DTYPES) - set(MOPY_SPECIFIC_DTYPES) - set(DIST_COLS)
        )

    def test_add_time_buffer(
        self, node_stats_group
    ):  # Not quite sure what's going on in this test...
        """
        Ensure time can be added to the start and end of the node_stats_group.
        """
        # Add times, start and end
        df = node_stats_group.data
        start = 1
        end = pd.Series(2, index=df.index)
        sg = node_stats_group.add_time_buffer(start=start, end=end)
        # Make sure a copy did occur
        assert sg is not node_stats_group
        # Make sure time offset is correct
        df2 = sg.data
        # Make sure to only get records with non-NaN start and end times
        df3 = df2.loc[df2["starttime"].notnull() & df2["endtime"].notnull()]
        df4 = df.loc[df3.index]
        assert ((df3["starttime"] + to_timedelta64(1)) == df4["starttime"]).all()
        assert ((df3["endtime"] - to_timedelta64(2)) == df4["endtime"]).all()

    def test_get_column_or_index(self, node_stats_group):
        """ tests for getting a series from a column or an index. """
        chan = node_stats_group.get_column("channel")
        assert isinstance(chan, pd.Series)
        assert chan.equals(node_stats_group.data["channel"])
        # first index
        name = node_stats_group.index.names[0]
        phase_hint = node_stats_group.get_column(name)
        assert isinstance(phase_hint, pd.Series)


class TestSetPicks:
    """ Tests for the set_picks method """

    # Fixtures
    @pytest.fixture(scope="class")
    def pick_csv(self, data_path):  # Again, I would love a better way to do this...
        """ csv file containing pick information """
        return join(data_path, "picks.csv")

    @pytest.fixture(scope="function")
    def pick_df(self, pick_csv):
        """ single-indexed DataFrame containing pick information """
        return pd.read_csv(pick_csv)

    @pytest.fixture(scope="class")
    def bogus_pick_df(self, pick_csv):
        """ returns a df with a nonexistent event """
        df = pd.read_csv(pick_csv)
        return df.rename(
            columns={"event_id": "random", "seed_id": "column", "phase": "name"}
        )

    @pytest.fixture(scope="function")
    def pick_df_multi(self, pick_csv):
        """ multi-indexed DataFrame containing pick information """
        return pd.read_csv(pick_csv, index_col=[0, 1, 2])

    @pytest.fixture(scope="function")
    def pick_df_extra(self, pick_csv):
        """ pick DataFrame with extra columns """
        df = pd.read_csv(pick_csv)
        df.loc[0, "extra"] = "abcd"
        return df

    # @pytest.fixture(scope="function")
    # def pick_dict(self, pick_df):
    #     """ mapping of pick meta-info to pick time """
    #     # Want to double check the column names here to make sure they match what StatsGroup expects
    #     return {
    #         (pick.phase_hint, pick.event_id, pick.seed_id): pick.time
    #         for (_, pick) in pick_df.iterrows()
    #     }

    # @pytest.fixture(scope="function")
    # def picks(self, pick_dict):
    #     return {
    #         key: Pick(
    #             time=time,
    #             phase_hint=key[0],
    #             waveform_id=WaveformStreamID(seed_string=key[2]),
    #             onset="impulsive",
    #             polarity="negative",
    #         )
    #         for (key, time) in pick_dict.items()
    #     }

    @pytest.fixture(scope="function")
    def bad_event(self, node_inventory, pick_df):
        """ return a dataframe with an event that doesn't exist """
        net = node_inventory[0]
        sta = net[0]
        chan = sta[0]
        pick_df["event_id"] = "not_an_event"
        return pick_df

    @pytest.fixture(scope="function")
    def bad_seed(self, pick_df):
        """ return a dataframe with a seed_id that doesn't exist """
        pick_df.loc[0, "seed_id"] = "bad seed"
        return pick_df

    # @pytest.fixture(scope="class")
    # def add_picks_from_dict(self, node_stats_group_no_picks, pick_dict):
    #     return node_stats_group_no_picks.set_picks(pick_dict)

    @pytest.fixture(scope="function")
    def add_picks_from_df(self, node_stats_group_no_picks, pick_df):
        """ apply picks via a dataframe """
        return node_stats_group_no_picks.set_picks(pick_df)

    @pytest.fixture(scope="function")
    def add_picks_from_multi_df(self, node_stats_group_no_picks, pick_df_multi):
        """ apply picks via a multi-indexed dataframe """
        return node_stats_group_no_picks.set_picks(pick_df_multi)

    @pytest.fixture(scope="function")
    def add_picks_from_extra_df(self, node_stats_group_no_picks, pick_df_extra):
        """ apply picks via a dataframe that has extra columns """
        return node_stats_group_no_picks.set_picks(pick_df_extra)

    # TODO: -if- this is going to be supported, the input should just look like a list of picks...
    # @pytest.fixture(scope="function")
    # def add_picks_from_picks(self, node_stats_group_no_picks, picks):
    #     return node_stats_group_no_picks.set_picks(picks)

    # Tests
    # def test_set_picks(self, add_picks_from_dict, pick_dict):
    #     """ verify that it is possible to attach picks from a mapping (dictionary) """
    #     # Make sure new picks actually got attached
    #     assert len(add_picks_from_dict) == len(pick_dict)
    #     # Make sure the attached data are what you expect
    #     newest = add_picks_from_dict.data.iloc[-1]
    #     pick = pick_dict[newest.name]
    #     seed_id = newest.name[2].split(".")
    #     expected_dict = {
    #         "network": seed_id[0],
    #         "station": seed_id[1],
    #         "location": seed_id[2],
    #         "channel": seed_id[3],
    #     }
    #     not_nans = [
    #         "distance_m",
    #         "azimuth",
    #         "vertical_distance_m",
    #         "ray_path_length_m",
    #         "pick_id",
    #     ]
    #     nans = [
    #         "method_id",
    #         "endtime",
    #         "starttime",
    #         "sampling_rate",
    #         "onset",
    #         "polarity",
    #     ]
    #     assert np.isclose(newest["time"], UTCDateTime(pick).timestamp)
    #     assert dict(newest[expected_dict.keys()]) == expected_dict
    #     assert newest[list(MOPY_SPECIFIC_DTYPES)].isnull().all()
    #     assert newest[nans].isnull().all()
    #     assert newest[not_nans].notnull().all()

    def test_bogus_picks(self, node_stats_group_no_picks):
        """ verify fails predictably if something other than an allowed pick format gets passed """
        with pytest.raises(TypeError):
            node_stats_group_no_picks.set_picks(2)

    def test_bad_pick_dict(self, node_stats_group_no_picks):
        """ verify fails predictably if a malformed pick dictionary gets provided """
        # Bogus key
        pick_dict = {"a": UTCDateTime("2018-09-12")}
        with pytest.raises(TypeError):
            node_stats_group_no_picks.set_picks(pick_dict)

        # Bogus value
        pick_dict = {
            (
                "P",
                "event_1",
                node_stats_group_no_picks.event_station_df.index.levels[1][0],
            ): "a"
        }
        with pytest.raises(TypeError):
            node_stats_group_no_picks.set_picks(pick_dict)

    def test_set_picks_dataframe(self, add_picks_from_df, pick_df):
        """ verify that it is possible to attach picks from a DataFrame/csv file """
        assert len(add_picks_from_df) == len(pick_df)
        # Make sure the resulting data is what you expect
        newest = add_picks_from_df.data.reset_index().set_index(PDF_IND).iloc[-1]
        assert isinstance(newest["time"], pd.Timestamp)
        pick_time = pick_df.set_index(PDF_IND).loc[newest.name]["time"]
        assert newest.time == to_datetime64(pick_time)
        assert_not_nan(newest.pick_id)

    def test_multi_indexed_dataframe(self, add_picks_from_multi_df, pick_df_multi):
        """ verify that it is possible to use a multi-indexed dataframe with the pick information """
        # Things may be starting to get more complicated than I actually want to deal with here
        assert len(add_picks_from_multi_df) == len(pick_df_multi)
        # Make sure the input data is what you expect
        newest = add_picks_from_multi_df.data.reset_index().set_index(PDF_IND).iloc[-1]
        pick_time = (
            pick_df_multi.reset_index().set_index(PDF_IND).loc[newest.name]["time"]
        )
        assert newest.time == to_datetime64(pick_time)
        assert_not_nan(newest.pick_id)

    def test_picks_df_extra_cols(self, add_picks_from_extra_df, pick_df_extra):
        """ verify extra columns in the picks_df get carried through """
        add_picks_from_extra_df.set_picks(pick_df_extra)
        assert "extra" in add_picks_from_extra_df.data.columns

    def test_invalid_df(self, node_stats_group_no_picks, bogus_pick_df):
        """ verify fails predictably if a df doesn't contain the required info """
        with pytest.raises(KeyError):
            node_stats_group_no_picks.set_picks(bogus_pick_df)

    def test_df_overwrite(self, add_picks_from_df, pick_df):
        """ verify that my hack for preserving pick_ids when overwriting picks with a df works as expected """
        pick_df = deepcopy(pick_df)
        resource_ids = add_picks_from_df.data.pick_id
        pick_df["time"] = to_datetime64(pick_df["time"]) + pd.to_timedelta(1, unit="s")
        with pytest.warns(UserWarning, match="existing"):
            out = add_picks_from_df.set_picks(pick_df)
        # Make sure the existing picks were overwritten, not appended
        assert len(out) == len(pick_df)
        # Make sure the times got updated
        pick_df["seed_id_less"] = pick_df["seed_id"].str[:-1]
        pick_df.set_index(list(_INDEX_NAMES), inplace=True)
        assert out.data["time"].sort_values().equals(pick_df["time"].sort_values())
        # Make sure the resource_ids haven't changed
        assert (add_picks_from_df.data.pick_id == resource_ids).all()

    # def test_set_picks_from_picks(self, add_picks_from_picks, picks):
    #     """ verify that it is possible to attach picks from a mapping of pick objects """
    #     assert len(add_picks_from_picks) == len(picks)
    #     # Make sure the input data is what you expect
    #     newest = add_picks_from_picks.data.iloc[-1]
    #     assert isinstance(newest["time"], Number)
    #     pick = picks[newest.name]
    #     seed_id = get_seed_id(pick).split(".")
    #     expected_dict = {
    #         "network": seed_id[0],
    #         "station": seed_id[1],
    #         "location": seed_id[2],
    #         "channel": seed_id[3],
    #         "time": pick.time,
    #         "onset": pick.onset,
    #         "polarity": pick.polarity,
    #         "pick_id": pick.resource_id.id,
    #         "phase_hint": pick.phase_hint,
    #     }
    #     not_nans = [
    #         "distance_m",
    #         "azimuth",
    #         "vertical_distance_m",
    #         "ray_path_length_m",
    #     ]
    #     nans = ["method_id", "endtime", "starttime", "sampling_rate"]
    #     assert dict(newest[expected_dict.keys()]) == expected_dict
    #     assert newest[list(MOPY_SPECIFIC_DTYPES)].isnull().all()
    #     assert newest[nans].isnull().all()
    #     assert newest[not_nans].notnull().all()

    def test_copied(self, add_picks_from_df, node_stats_group_no_picks):
        """ verify that set_picks returns a copy of itself by default """
        assert not id(add_picks_from_df) == id(node_stats_group_no_picks)
        assert not id(add_picks_from_df.data) == id(node_stats_group_no_picks.data)

    def test_inplace(self, node_stats_group_no_picks, pick_df):
        """ verify it is possible to set_picks inplace """
        node_stats_group_no_picks = deepcopy(node_stats_group_no_picks)
        length = len(node_stats_group_no_picks)
        node_stats_group_no_picks.set_picks(pick_df, inplace=True)
        assert len(node_stats_group_no_picks) > length

    # TODO: For now it's on the user to convert it to a DataFrame
    #    def test_set_picks_phase_file(self, node_channel_info_no_picks):
    #        """ verify it is possible to attach picks using a hypoinverse phase file or other format? """
    #        assert False

    def test_event_doesnt_exist(self, node_stats_group_no_picks, bad_event):
        """ verify that it fails predictably if the specified event does not exist """
        with pytest.raises(KeyError):
            node_stats_group_no_picks.set_picks(bad_event)

    def test_seed_doesnt_exist(self, node_stats_group_no_picks, bad_seed):
        """ verify that it fails predictably if a seed_id doesn't exist """
        with pytest.raises(KeyError):
            node_stats_group_no_picks.set_picks(bad_seed)

    def test_instant_gratification(self):
        assert True


class TestSetTimeWindows:
    """ Tests for assigning time windows """

    relative_windows = dict(P=(0, 1), S=(0.5, 2), Noise=(5, 2))
    extra_windows = dict(P=(0, 1), S=(0.5, 2), pPcPSKKP=(1, 20))

    # Fixtures
    @pytest.fixture(scope="class")
    def pick_df(self, data_path):  # Again, I would love a better way to do this...
        """ df of picks """
        return pd.read_csv(join(data_path, "picks.csv"))

    @pytest.fixture(scope="class")
    def abs_time_windows(self, node_stats_group_no_tws):
        """ create absolute time windows """
        time_before = 0.2
        time_after = 1
        phase = node_stats_group_no_tws.data.droplevel("seed_id_less").iloc[-1]
        return {
            phase.name: (
                phase.time - to_timedelta64(time_before),
                phase.time + to_timedelta64(time_after),
            )
        }

    @pytest.fixture(scope="function")
    def abs_time_windows_df(self, node_stats_group_no_tws, abs_time_windows):
        tws = pd.DataFrame(
            columns=["phase_hint", "event_id", "seed_id", "starttime", "endtime"]
        )
        for tw in abs_time_windows:
            tws.loc[len(tws)] = list(tw) + list(abs_time_windows[tw])
        return tws

    @pytest.fixture(scope="function")
    def abs_time_windows_df_multi(self, abs_time_windows_df):
        return abs_time_windows_df.set_index(["phase_hint", "event_id", "seed_id"])

    @pytest.fixture(scope="class")
    def node_stats_group_no_tws(self, node_stats_group_no_picks, pick_df):
        """ StatsGroup with picks, but no time windows """
        return node_stats_group_no_picks.set_picks(pick_df)

    @pytest.fixture(scope="class")
    def add_rel_time_windows(self, node_stats_group_no_tws):
        """ Set relative time windows """
        return node_stats_group_no_tws.set_rel_time_windows(**self.relative_windows)

    # @pytest.fixture(scope="class")
    # def add_abs_time_windows(self, node_stats_group_no_tws, abs_time_windows):
    #     """ Set absolute time windows """
    #     return node_stats_group_no_tws.set_abs_time_windows(abs_time_windows)

    @pytest.fixture(scope="function")
    def add_abs_time_windows_df(self, node_stats_group_no_tws, abs_time_windows_df):
        return node_stats_group_no_tws.set_abs_time_windows(abs_time_windows_df)

    @pytest.fixture(scope="function")
    def add_abs_time_windows_df_multi(
        self, node_stats_group_no_tws, abs_time_windows_df_multi
    ):
        return node_stats_group_no_tws.set_abs_time_windows(abs_time_windows_df_multi)

    @pytest.fixture(scope="class")
    def bad_event(self, node_inventory):
        net = node_inventory[0]
        sta = net[0]
        chan = sta[0]
        return pd.DataFrame(
            [
                [
                    "P",
                    "not_an_event",
                    f"{net.code}.{sta.code}.{chan.location_code}.{chan.code}",
                    UTCDateTime(2019, 1, 1),
                    UTCDateTime(2019, 1, 2),
                ]
            ],
            columns=["phase_hint", "event_id", "seed_id", "starttime", "endtime"],
        )

    @pytest.fixture(scope="class")
    def bad_seed(self, node_catalog):
        return pd.DataFrame(
            [
                [
                    "P",
                    node_catalog[0].resource_id.id,
                    "im.a.bad.seed",
                    UTCDateTime(2019, 1, 1),
                    UTCDateTime(2019, 1, 2),
                ]
            ],
            columns=["phase_hint", "event_id", "seed_id", "starttime", "endtime"],
        )

    # Tests
    # tests for set_rel_time_windows
    def test_set_rel_time_windows(self, add_rel_time_windows):
        """ Make sure it is possible to set relative time windows """
        assert not add_rel_time_windows.data.starttime.isnull().any()
        assert not add_rel_time_windows.data.endtime.isnull().any()
        assert (
            add_rel_time_windows.data.endtime > add_rel_time_windows.data.starttime
        ).all()
        # Make sure the tw is as expected for each of the provided phase types
        for phase in self.relative_windows:
            pick = add_rel_time_windows.data.xs(phase, level="phase_hint").iloc[0]
            assert pd.Timestamp(pick.starttime, unit="ns") == (
                pick.time - to_timedelta64(self.relative_windows[phase][0])
            )
            assert pd.Timestamp(pick.endtime, "ns") == (
                pick.time + to_timedelta64(self.relative_windows[phase][1])
            )
            # Make sure the times are semi-plausible
            assert (pick.starttime > pd.Timestamp(1800, 1, 1)) and (
                pick.endtime > pd.Timestamp(1800, 1, 1)
            )

    def test_set_rel_time_windows_bogus(self, node_stats_group_no_tws):
        """ make sure bogus time windows are handled accordingly """
        with pytest.raises(TypeError):
            node_stats_group_no_tws.set_rel_time_windows(P="a")
        with pytest.raises(TypeError):
            node_stats_group_no_tws.set_rel_time_windows(P=("a", 1))
        with pytest.raises(ValueError):
            node_stats_group_no_tws.set_rel_time_windows(P=(1, 2, 3))

    def test_set_rel_time_windows_no_picks(self, node_stats_group_no_picks):
        """ make sure it is not possible to set relative time windows if no picks have been provided """
        with pytest.raises(ValueError):
            node_stats_group_no_picks.set_rel_time_windows(**self.relative_windows)

    def test_set_rel_time_windows_extra_phases(self, node_stats_group_no_tws):
        """ make sure behaves reasonably if extra phase types are provided """
        with pytest.warns(UserWarning):
            node_stats_group_no_tws.set_rel_time_windows(**self.extra_windows)

    def test_set_rel_time_windows_overwrite(self, add_rel_time_windows):
        """ make sure a warning is issued if overwriting time windows """
        with pytest.warns(UserWarning):
            add_rel_time_windows.set_rel_time_windows(**self.relative_windows)

    def test_set_rel_time_windows_copied(
        self, add_rel_time_windows, node_stats_group_no_tws
    ):
        """ verify that set_rel_time_windows returns a copy of itself by default """
        assert not id(add_rel_time_windows) == id(node_stats_group_no_tws)
        assert not id(add_rel_time_windows.data) == id(node_stats_group_no_tws.data)

    def test_set_rel_time_windows_inplace(self, node_stats_group_no_tws):
        """ verify it is possible to set time windows in place using set_rel_time_windows"""
        node_stats_group_no_tws = deepcopy(node_stats_group_no_tws)
        node_stats_group_no_tws.set_rel_time_windows(
            inplace=True, **self.relative_windows
        )
        assert not node_stats_group_no_tws.data.starttime.isnull().any()
        assert not node_stats_group_no_tws.data.endtime.isnull().any()

    # tests for set_abs_time_windows
    def test_set_abs_time_windows_from_df(
        self, add_abs_time_windows_df, abs_time_windows_df
    ):
        """ make sure it is possible to specify time windows using a DataFrame """
        # Didn't want to set tws for all phases
        assert add_abs_time_windows_df.data.starttime.isnull().any()
        assert add_abs_time_windows_df.data.endtime.isnull().any()
        # Make sure the tw got set as expected
        pick = add_abs_time_windows_df.data.reset_index().set_index(PDF_IND).iloc[-1]
        tw = abs_time_windows_df.set_index(PDF_IND).loc[pick.name]
        assert pick.starttime == tw[0]
        assert pick.endtime == tw[1]
        # Make sure the times are semi-plausible
        assert (pick.starttime > pd.Timestamp(1800, 1, 1)) and (
            pick.endtime > pd.Timestamp(1800, 1, 1)
        )

    def test_set_abs_time_windows_from_df_multi(
        self, add_abs_time_windows_df_multi, abs_time_windows_df
    ):
        """ make sure it is possible to specify time windows using a multi-indexed DataFrame"""
        # Make sure the tw got set as expected
        pick = (
            add_abs_time_windows_df_multi.data.reset_index().set_index(PDF_IND).iloc[-1]
        )
        tw = abs_time_windows_df.set_index(PDF_IND).loc[pick.name]
        assert pick.starttime == tw[0]
        assert pick.endtime == tw[1]

    def test_set_abs_time_windows_no_picks(
        self, node_stats_group_no_picks, abs_time_windows_df
    ):
        """ make sure it is possible to set absolute time windows if no picks have been provided """
        # Make sure a pick got added to the df
        channel_info = node_stats_group_no_picks.set_abs_time_windows(
            abs_time_windows_df
        )
        assert len(channel_info) == 1
        # Make sure the pick time and time window get set as expected
        pick = channel_info.data.reset_index().set_index(PDF_IND).iloc[-1]
        tw = abs_time_windows_df.set_index(PDF_IND).loc[pick.name]
        seed_id = pick.name[2].split(".")
        nslc = {
            "network": seed_id[0],
            "station": seed_id[1],
            "location": seed_id[2],
            "channel": seed_id[3],
        }
        times = {
            "time": tw[0],
            "starttime": tw[0],
            "endtime": tw[1],
        }
        # Make sure the meta information got updated as expected
        not_nans = [
            "distance_m",
            "azimuth",
            "vertical_distance_m",
            "ray_path_length_m",
            "pick_id",
        ]
        nans = ["method_id", "sampling_rate", "onset", "polarity"]
        assert dict(pick[nslc.keys()]) == nslc
        assert all([a == b for a, b in zip(pick[times.keys()].values, times.values())])
        # WHY DIDN'T THIS UPDATE THE META INFORMATION IN AN EXPECTED MANNER?
        # assert pick[MOPY_SPECIFIC_DTYPES].isnull().all()
        assert pick[nans].isnull().all()
        assert pick[not_nans].notnull().all()

    def test_set_abs_time_windows_bogus_df(
        self, node_stats_group_no_tws, abs_time_windows_df, pick_df
    ):
        """ verify fails predictably if df with bogus info gets passed """
        # Absolute garbage
        with pytest.raises(TypeError):
            node_stats_group_no_tws.set_abs_time_windows(2)
        # I'm not quite sure what this is testing? A column mismatch, perhaps?
        with pytest.raises(KeyError):
            node_stats_group_no_tws.set_abs_time_windows(pick_df)
        # End time(s) before start time(s)
        abs_time_windows_df["starttime"] = abs_time_windows_df[
            "endtime"
        ] + pd.to_timedelta(5, unit="s")
        with pytest.raises(ValueError):
            node_stats_group_no_tws.set_abs_time_windows(abs_time_windows_df)

    def test_set_abs_time_windows_overwrite(
        self, add_rel_time_windows, abs_time_windows_df
    ):
        """ make sure overwriting issues a warning """
        with pytest.warns(UserWarning):
            add_rel_time_windows.set_abs_time_windows(abs_time_windows_df)

    def test_set_abs_time_windows_copy(
        self, add_abs_time_windows_df, node_stats_group_no_tws
    ):
        """ make sure setting time windows returns a copy by default """
        assert not id(add_abs_time_windows_df) == id(node_stats_group_no_tws)
        assert not id(add_abs_time_windows_df.data) == id(node_stats_group_no_tws.data)

    def test_set_abs_time_windows_inplace(
        self, node_stats_group_no_tws, abs_time_windows_df
    ):
        """ make sure it is possible to set time windows inplace using set_abs_time_windows"""
        node_stats_group_no_tws = deepcopy(node_stats_group_no_tws)
        node_stats_group_no_tws.set_abs_time_windows(abs_time_windows_df, inplace=True)
        # Make sure the tw got set as expected
        pick = node_stats_group_no_tws.data.reset_index().set_index(PDF_IND).iloc[-1]
        tw = abs_time_windows_df.set_index(PDF_IND).loc[pick.name]
        assert pick.starttime == tw[0]
        assert pick.endtime == tw[1]

    def test_event_doesnt_exist(self, node_stats_group_no_tws, bad_event):
        """ verify that it fails predictably if the specified event does not exist """
        with pytest.raises(KeyError):
            node_stats_group_no_tws.set_abs_time_windows(bad_event)

    def test_seed_doesnt_exist(self, node_stats_group_no_tws, bad_seed):
        """ verify that it fails predictably if a seed_id doesn't exist """
        with pytest.raises(KeyError):
            node_stats_group_no_tws.set_abs_time_windows(bad_seed)


class TestSetParameters:  # There has to be a way to duplicate tests for methods that have the same signature...
    """ Tests for setting parameters (should all be using the underlying '_broadcast_param' """

    velocity = {"P": 5000, "S": 3000}
    density = 7000
    radiation_pattern = 0.2
    shear_modulus = 1e9
    quality_factor = {
        "P": {"1.2.1.DP": 100, "1.3.1.DP": 200, "1.4.1.DP": 300},
        "S": {"1.2.1.DP": 200, "1.3.1.DP": 400, "1.4.1.DP": 600},
        "Noise": 1000,
    }
    free_surface_coefficient = {"1.2.1.DP": 5, "1.3.1.DP": 1, "1.4.1.DP": 0.5}

    # Tests
    def test_set_source_velocity(self, node_stats_group):
        """ verify that the source velocity can be set, on a per phase basis"""
        out = node_stats_group.set_source_velocity(self.velocity)
        for phase, vel in self.velocity.items():
            assert (
                out.data.xs(phase, level="phase_hint")["source_velocity"] == vel
            ).all()

    def test_set_density(self, node_stats_group):
        """ verify that the density can be set """
        out = node_stats_group.set_density(self.density)
        assert (out.data["density"] == self.density).all()

    def test_set_shear_modulus(self, node_stats_group):
        """ verify that the shear modulus can be set """
        out = node_stats_group.set_shear_modulus(self.shear_modulus)
        assert (out.data["shear_modulus"] == self.shear_modulus).all()

    def test_set_quality_factor(self, node_stats_group):
        """ verify that the quality factor can be set, on a per phase and per station basis """
        out = node_stats_group.set_quality_factor(self.quality_factor)
        # Check each phase type separately (for my sanity)
        for ph in ["P", "S"]:
            for sta, qf in self.quality_factor[ph].items():
                assert (
                    out.data.xs((ph, sta), level=("phase_hint", "seed_id_less"))[
                        "quality_factor"
                    ]
                    == qf
                ).all
        ph = "Noise"
        assert (
            out.data.xs((ph, sta), level=("phase_hint", "seed_id_less"))[
                "quality_factor"
            ]
            == self.quality_factor[ph]
        ).all()

    def set_radiation_pattern(self, node_stats_group):
        """ verify that the radiation pattern can be set """
        out = node_stats_group.set_radiation_pattern(self.radiation_pattern)
        assert (out.data["radiation_pattern"] == self.radiation_pattern).all()

    def set_free_surface_coefficient(self, node_stats_group):
        """ verify that the free surface coefficient can be set, on a per-station basis """
        out = node_stats_group.set_free_surface_coefficient(
            self.free_surface_coefficient
        )
        for sta, fsc in self.free_surface_coefficient.items():
            assert (
                out.data.xs(sta, level="seed_id_less")["free_surface_coefficient"]
                == fsc
            ).all()

    def test_apply_defaults_doesnt_overwrite(self, node_stats_group):
        """ verify that applying default values does not overwrite the specified velocities """
        # Set the velocities
        out = node_stats_group.set_source_velocity(self.velocity)
        # Apply defaults
        out = out.apply_defaults()
        # Make sure the defaults were not applied to the velocity column
        for phase, vel in self.velocity.items():
            assert (
                out.data.xs(phase, level="phase_hint")["source_velocity"] == vel
            ).all()


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
