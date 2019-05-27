"""
Tests for channel information, specifically creating dataframes from it.
"""
from __future__ import annotations

from copy import deepcopy
from os.path import join
from numbers import Number

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


# --- Constants
sensible_defaults = {
    'velocity': 1800.,
    'radiation_coefficient': 0.6,
    'quality_factor': 400,
    'density': 3000,
    'shear_modulus': 2200000000,
    'free_surface_coefficient': 2.0}


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
    @pytest.fixture(scope="class")
    def pick_csv(self, data_path):  # Again, I would love a better way to do this...
        """ csv file containing pick information """
        return join(data_path, "picks.csv")

    @pytest.fixture(scope="class")
    def pick_df(self, pick_csv):
        """ single-indexed DataFrame containing pick information """
        return pd.read_csv(pick_csv)

    @pytest.fixture(scope="class")
    def bogus_pick_df(self, pick_csv):
        df = pd.read_csv(pick_csv)
        return df.rename(
            columns={"event_id": "random", "seed_id": "column", "phase": "name"}
        )

    @pytest.fixture(scope="class")
    def pick_df_multi(self, pick_csv):
        """ multi-indexed DataFrame containing pick information """
        return pd.read_csv(pick_csv, index_col=[0, 1, 2])

    @pytest.fixture(scope="class")
    def pick_df_extra(self, pick_csv):
        """ pick DataFrame with extra columns """
        df = pd.read_csv(pick_csv)
        df.loc[0, "extra"] = "abcd"
        return df

    @pytest.fixture(scope="class")
    def pick_dict(self, pick_df):
        """ mapping of pick meta-info to pick time """
        # Want to double check the column names here to make sure they match what ChannelInfo expects
        return {
            (pick.phase_hint, pick.event_id, pick.seed_id): pick.time
            for (_, pick) in pick_df.iterrows()
        }

    @pytest.fixture(scope="class")
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

    @pytest.fixture(scope="class")
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

    @pytest.fixture(scope="class")
    def bad_seed(self, node_catalog):
        return {
            ("P", node_catalog[0].resource_id.id, "bad seed"): UTCDateTime(2019, 1, 1)
        }

    @pytest.fixture(scope="class")
    def add_picks_from_dict(self, node_channel_info_no_picks, pick_dict):
        return node_channel_info_no_picks.set_picks(pick_dict)

    @pytest.fixture(scope="class")
    def add_picks_from_df(self, node_channel_info_no_picks, pick_df):
        return node_channel_info_no_picks.set_picks(pick_df)

    @pytest.fixture(scope="class")
    def add_picks_from_multi_df(self, node_channel_info_no_picks, pick_df_multi):
        return node_channel_info_no_picks.set_picks(pick_df_multi)

    @pytest.fixture(scope="class")
    def add_picks_from_extra_df(self, node_channel_info_no_picks, pick_df_extra):
        return node_channel_info_no_picks.set_picks(pick_df_extra)

    @pytest.fixture(scope="class")
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
            'network': seed_id[0],
            'station': seed_id[1],
            'location': seed_id[2],
            'channel': seed_id[3],
        }
        not_nans = ['distance', 'azimuth', 'horizontal_distance', 'depth_distance', 'ray_path_length', 'spreading_coefficient', 'pick_id']
        nans = ['method_id', 'endtime', 'starttime', 'sampling_rate', 'onset', 'polarity']
        assert np.isclose(newest['time'], UTCDateTime(pick).timestamp)
        assert dict(newest[expected_dict.keys()]) == expected_dict
        assert np.isclose(newest[sensible_defaults.keys()].values.astype(np.float64),
                          list(sensible_defaults.values())).all()  # The dtypes for the picks needs to be overridden somewhere...
        assert newest[nans].isnull().all()
        assert newest[not_nans].notnull().all()

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
        assert isinstance(newest['time'], Number)
        pick_time = pick_df.set_index(["phase_hint", "event_id", "seed_id"]).loc[newest.name].time
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
        """ verify extra columns in the picks_df get carried through """
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
        assert len(add_picks_from_picks) == len(picks)
        # Make sure the input data is what you expect
        newest = add_picks_from_picks.data.iloc[-1]
        assert isinstance(newest['time'], Number)
        pick = picks[newest.name]
        seed_id = get_seed_id(pick).split(".")
        expected_dict = {
            'network': seed_id[0],
            'station': seed_id[1],
            'location': seed_id[2],
            'channel': seed_id[3],
            'time': pick.time,
            'onset': pick.onset,
            'polarity': pick.polarity,
            'pick_id': pick.resource_id.id,
            'phase_hint': pick.phase_hint,
}
        not_nans = ['distance', 'azimuth', 'horizontal_distance', 'depth_distance', 'ray_path_length',
                    'spreading_coefficient']
        nans = ['method_id', 'endtime', 'starttime', 'sampling_rate']
        assert dict(newest[expected_dict.keys()]) == expected_dict
        assert np.isclose(newest[sensible_defaults.keys()].values.astype(np.float64),
                          list(sensible_defaults.values())).all()  # The dtypes for the picks needs to be overridden somewhere...
        assert newest[nans].isnull().all()
        assert newest[not_nans].notnull().all()

    def test_copied(self, add_picks_from_dict, node_channel_info_no_picks):
        """ verify that set_picks returns a copy of itself by default """
        assert not id(add_picks_from_dict) == id(node_channel_info_no_picks)
        assert not id(add_picks_from_dict.data) == id(node_channel_info_no_picks.data)

    def test_inplace(self, node_channel_info_no_picks, pick_dict):
        """ verify it is possible to set_picks inplace """
        node_channel_info_no_picks = deepcopy(node_channel_info_no_picks)
        length = len(node_channel_info_no_picks)
        node_channel_info_no_picks.set_picks(pick_dict, inplace=True)
        assert len(node_channel_info_no_picks) > length

    # TODO: For now it's on the user to convert it to a DataFrame
    #    def test_set_picks_phase_file(self, node_channel_info_no_picks):
    #        """ verify it is possible to attach picks using a hypoinverse phase file or other format? """
    #        assert False

    def test_event_doesnt_exist(self, node_channel_info_no_picks, bad_event):
        """ verify that it fails predictably if the specified event does not exist """
        with pytest.raises(KeyError):
            node_channel_info_no_picks.set_picks(bad_event)

    def test_seed_doesnt_exist(self, node_channel_info_no_picks, bad_seed):
        """ verify that it fails predictably if a seed_id doesn't exist """
        with pytest.raises(KeyError):
            node_channel_info_no_picks.set_picks(bad_seed)

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
    def abs_time_windows(self, node_channel_info_no_tws):
        """ create absolute time windows """
        time_before = 0.2
        time_after = 1
        phase = node_channel_info_no_tws.data.iloc[-1]
        # Pass times as strings to make sure they can be correctly converted to timestamps
        return {phase.name: (str(phase.time - time_before), str(phase.time + time_after))}

    @pytest.fixture(scope="class")
    def abs_time_windows_df(self, abs_time_windows):
        tws = pd.DataFrame(columns=["phase_hint", "event_id", "seed_id", "starttime", "endtime"])
        for tw in abs_time_windows:
            tws.loc[len(tws)] = list(tw) + list(abs_time_windows[tw])
        return tws

    @pytest.fixture(scope="class")
    def abs_time_windows_df_multi(self, abs_time_windows_df):
        return abs_time_windows_df.set_index(["phase_hint", "event_id", "seed_id"])

    @pytest.fixture(scope="class")
    def node_channel_info_no_tws(self, node_channel_info_no_picks, pick_df):
        """ ChannelInfo with picks, but no time windows """
        return node_channel_info_no_picks.set_picks(pick_df)

    @pytest.fixture(scope='class')
    def add_rel_time_windows(self, node_channel_info_no_tws):
        """ Set relative time windows """
        return node_channel_info_no_tws.set_rel_time_windows(**self.relative_windows)

    @pytest.fixture(scope='class')
    def add_abs_time_windows(self, node_channel_info_no_tws, abs_time_windows):
        """ Set absolute time windows """
        return node_channel_info_no_tws.set_abs_time_windows(abs_time_windows)

    @pytest.fixture(scope='class')
    def add_abs_time_windows_df(self, node_channel_info_no_tws, abs_time_windows_df):
        return node_channel_info_no_tws.set_abs_time_windows(abs_time_windows_df)

    @pytest.fixture(scope='class')
    def add_abs_time_windows_df_multi(self, node_channel_info_no_tws, abs_time_windows_df_multi):
        return node_channel_info_no_tws.set_abs_time_windows(abs_time_windows_df_multi)

    @pytest.fixture(scope="class")
    def bad_event(self, node_inventory):
        net = node_inventory[0]
        sta = net[0]
        chan = sta[0]
        return {("P", "not_an_event", f"{net.code}.{sta.code}.{chan.location_code}.{chan.code}"):
                (UTCDateTime(2019, 1, 1), UTCDateTime(2019, 1, 2))}

    @pytest.fixture(scope="class")
    def bad_seed(self, node_catalog):
        return {("P", node_catalog[0].resource_id.id, "bad seed"): (UTCDateTime(2019, 1, 1), UTCDateTime(2019, 1, 2))}

    # Tests
    # tests for set_rel_time_windows
    def test_set_rel_time_windows(self, add_rel_time_windows):
        """ Make sure it is possible to set relative time windows """
        assert not add_rel_time_windows.data.starttime.isnull().any()
        assert not add_rel_time_windows.data.endtime.isnull().any()
        assert (add_rel_time_windows.data.endtime > add_rel_time_windows.data.starttime).all()
        # Make sure the tw is as expected for each of the provided phase types
        for phase in self.relative_windows:
            pick = add_rel_time_windows.data.xs(phase, level="phase_hint").iloc[0]
            assert np.isclose(pick.starttime, pick.time - self.relative_windows[phase][0])
            assert np.isclose(pick.endtime, pick.time + self.relative_windows[phase][1])

    def test_set_rel_time_windows_bogus(self, node_channel_info_no_tws):
        """ make sure bogus time windows are handled accordingly """
        with pytest.raises(TypeError):
            node_channel_info_no_tws.set_rel_time_windows(P="a")
        with pytest.raises(TypeError):
            node_channel_info_no_tws.set_rel_time_windows(P=("a", 1))
        with pytest.raises(ValueError):
            node_channel_info_no_tws.set_rel_time_windows(P=(1, 2, 3))

    def test_set_rel_time_windows_no_picks(self, node_channel_info_no_picks):
        """ make sure it is not possible to set relative time windows if no picks have been provided """
        with pytest.raises(ValueError):
            node_channel_info_no_picks.set_rel_time_windows(**self.relative_windows)

    def test_set_rel_time_windows_extra_phases(self, node_channel_info_no_tws):
        """ make sure behaves reasonably if extra phase types are provided """
        with pytest.warns(UserWarning):
            node_channel_info_no_tws.set_rel_time_windows(**self.extra_windows)

    def test_set_rel_time_windows_overwrite(self, add_rel_time_windows):
        """ make sure a warning is issued if overwriting time windows """
        with pytest.warns(UserWarning):
            add_rel_time_windows.set_rel_time_windows(**self.relative_windows)

    def test_set_rel_time_windows_copied(self, add_rel_time_windows, node_channel_info_no_tws):
        """ verify that set_rel_time_windows returns a copy of itself by default """
        assert not id(add_rel_time_windows) == id(node_channel_info_no_tws)
        assert not id(add_rel_time_windows.data) == id(node_channel_info_no_tws.data)

    def test_set_rel_time_windows_inplace(self, node_channel_info_no_tws):
        """ verify it is possible to set time windows in place using set_rel_time_windows"""
        node_channel_info_no_tws = deepcopy(node_channel_info_no_tws)
        node_channel_info_no_tws.set_rel_time_windows(inplace=True, **self.relative_windows)
        assert not node_channel_info_no_tws.data.starttime.isnull().any()
        assert not node_channel_info_no_tws.data.endtime.isnull().any()

    # tests for set_abs_time_windows
    def test_set_abs_time_windows(self, add_abs_time_windows, abs_time_windows):
        """ make sure it is possible to set absolute time windows """
        # Didn't want to set tws for all phases
        assert add_abs_time_windows.data.starttime.isnull().any()
        assert add_abs_time_windows.data.endtime.isnull().any()
        # Make sure the tw got set as expected
        pick = add_abs_time_windows.data.iloc[-1]
        tw = abs_time_windows[pick.name]
        assert np.isclose(pick.starttime, UTCDateTime(tw[0]).timestamp)
        assert np.isclose(pick.endtime, UTCDateTime(tw[1]).timestamp)

    def test_set_abs_time_windows_no_picks(self, node_channel_info_no_picks, abs_time_windows):
        """ make sure it is possible to set absolute time windows if no picks have been provided """
        # Make sure a pick got added to the df
        #breakpoint()
        channel_info = node_channel_info_no_picks.set_abs_time_windows(abs_time_windows)
        assert len(channel_info) == 1
        # Make sure the pick time and time window get set as expected
        pick = channel_info.data.iloc[-1]
        tw = abs_time_windows[pick.name]
        seed_id = pick.name[2].split(".")
        nslc = {
            'network': seed_id[0],
            'station': seed_id[1],
            'location': seed_id[2],
            'channel': seed_id[3]
        }
        times = {
            'time': UTCDateTime(tw[0]).timestamp,
            'starttime': UTCDateTime(tw[0]).timestamp,
            'endtime': UTCDateTime(tw[1]).timestamp,
        }
        # Make sure the meta information got updated as expected
        not_nans = ['distance', 'azimuth', 'horizontal_distance', 'depth_distance', 'ray_path_length',
                    'spreading_coefficient', 'pick_id']
        nans = ['method_id', 'sampling_rate', 'onset', 'polarity']
        assert dict(pick[nslc.keys()]) == nslc
        assert np.isclose(pick[times.keys()].values.astype(np.float64), list(times.values())).all()  # The dtypes for the picks needs to be overridden somewhere...
        assert np.isclose(pick[sensible_defaults.keys()].values.astype(np.float64),
                          list(sensible_defaults.values())).all()  # The dtypes for the picks needs to be overridden somewhere...
        assert pick[nans].isnull().all()
        assert pick[not_nans].notnull().all()

    def test_set_abs_time_windows_from_df(self, add_abs_time_windows_df, abs_time_windows):
        """ make sure it is possible to specify time windows using a DataFrame """
        # Make sure the tw got set as expected
        pick = add_abs_time_windows_df.data.iloc[-1]
        tw = abs_time_windows[pick.name]
        assert np.isclose(pick.starttime, UTCDateTime(tw[0]).timestamp)
        assert np.isclose(pick.endtime, UTCDateTime(tw[1]).timestamp)

    def test_set_abs_time_windows_from_df_multi(self, add_abs_time_windows_df_multi, abs_time_windows):
        """ make sure it is possible to specify time windows using a multi-indexed DataFrame"""
        # Make sure the tw got set as expected
        pick = add_abs_time_windows_df_multi.data.iloc[-1]
        tw = abs_time_windows[pick.name]
        assert np.isclose(pick.starttime, UTCDateTime(tw[0]).timestamp)
        assert np.isclose(pick.endtime, UTCDateTime(tw[1]).timestamp)

    def test_set_abs_time_windows_bogus(self, node_channel_info_no_tws):
        """ verify fails predictably if something other than an allowed tw format gets passed """
        with pytest.raises(TypeError):
            node_channel_info_no_tws.set_abs_time_windows(2)
        # Bogus key
        tws = {"a": (UTCDateTime("2018-09-12"), UTCDateTime("2018-09-13"))}
        with pytest.raises(TypeError):
            node_channel_info_no_tws.set_abs_time_windows(tws)
        # Bogus value
        ind = node_channel_info_no_tws.data.iloc[-1].name
        tws = {ind: 2}
        with pytest.raises(TypeError):
            node_channel_info_no_tws.set_abs_time_windows(tws)
        # End time before start time
        tws = {ind: (UTCDateTime("2018-09-12"), UTCDateTime("2018-09-11"))}
        with pytest.raises(ValueError):
            node_channel_info_no_tws.set_abs_time_windows(tws)

    def test_set_abs_time_windows_bogus_df(self, node_channel_info_no_tws, pick_df):
        """ verify fails predictably if df with bogus info gets passed """
        with pytest.raises(KeyError):
            node_channel_info_no_tws.set_abs_time_windows(pick_df)

    def test_set_abs_time_windows_overwrite(self, add_rel_time_windows, abs_time_windows):
        """ make sure overwriting issues a warning """
        with pytest.warns(UserWarning):
            add_rel_time_windows.set_abs_time_windows(abs_time_windows)

    def test_set_abs_time_windows_copy(self, add_abs_time_windows, node_channel_info_no_tws):
        """ make sure setting time windows returns a copy by default """
        assert not id(add_abs_time_windows) == id(node_channel_info_no_tws)
        assert not id(add_abs_time_windows.data) == id(node_channel_info_no_tws.data)

    def test_set_abs_time_windows_inplace(self, node_channel_info_no_tws, abs_time_windows):
        """ make sure it is possible to set time windows inplace using set_abs_time_windows"""
        node_channel_info_no_tws = deepcopy(node_channel_info_no_tws)
        node_channel_info_no_tws.set_abs_time_windows(abs_time_windows, inplace=True)
        # Make sure the tw got set as expected
        pick = node_channel_info_no_tws.data.iloc[-1]
        tw = abs_time_windows[pick.name]
        assert np.isclose(pick.starttime, UTCDateTime(tw[0]).timestamp)
        assert np.isclose(pick.endtime, UTCDateTime(tw[1]).timestamp)

    def test_event_doesnt_exist(self, node_channel_info_no_tws, bad_event):
        """ verify that it fails predictably if the specified event does not exist """
        with pytest.raises(KeyError):
            node_channel_info_no_tws.set_abs_time_windows(bad_event)

    def test_seed_doesnt_exist(self, node_channel_info_no_tws, bad_seed):
        """ verify that it fails predictably if a seed_id doesn't exist """
        with pytest.raises(KeyError):
            node_channel_info_no_tws.set_abs_time_windows(bad_seed)


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
