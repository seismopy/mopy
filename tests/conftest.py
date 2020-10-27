"""
pytest configuration for obsplus
"""
from __future__ import annotations

from os.path import join, dirname, abspath
from pathlib import Path

import obsplus
import obspy
import pytest
from obsplus.utils import get_reference_time
from obspy.core.event import Catalog, Event, ResourceIdentifier
from obspy.signal.invsim import corn_freq_2_paz

import mopy
import mopy.constants
from mopy import SpectrumGroup

TEST_PATH = abspath(dirname(__file__))
# path to the package directory
PKG_PATH = dirname(TEST_PATH)
# path to the test data directory
TEST_DATA_PATH = join(TEST_PATH, "data")
# test data cache
TEST_DATA_CACHE = join(TEST_DATA_PATH, "cached")


@pytest.fixture(scope="session", autouse=True)
def data_path():  # If there is a more eloquent way to do this, I'm all ears
    """ Return the path of the test datasets """
    return TEST_DATA_PATH


@pytest.fixture(scope="session", autouse=True)
def turn_on_debugging():
    """ Set the global debug flag to True. """
    mopy.constants.DEBUG = True


# --- Fixtures for crandall canyon DS


@pytest.fixture(scope="session")
def crandall_ds():
    """ Load the crandall canyon dataset """
    return obsplus.load_dataset("crandall_test")


@pytest.fixture(scope="session")
def crandall_catalog(crandall_ds):
    """ Return the crandall catalog. Add one P amplitude first. """
    cat = crandall_ds.event_client.get_events()
    # create dict of origin times/eids
    ot_id = {get_reference_time(eve).timestamp: eve for eve in cat}
    min_time = min(ot_id)
    event = ot_id[min_time]
    # add noise time amplitudes for a few channels (just so noise spectra
    path = Path(TEST_DATA_PATH) / "crandall_noise_picks.xml"
    noisy_cat = obspy.read_events(str(path), "quakeml")
    event.picks.extend(noisy_cat[0].picks)
    event.amplitudes.extend(noisy_cat[0].amplitudes)
    return cat


@pytest.fixture(scope="session")
def crandall_event(crandall_catalog):
    """ Return the fore-shock of the crandall collapse. """
    endtime = obspy.UTCDateTime("2007-08-06T08")
    cat = crandall_catalog.get_events(endtime=endtime)
    assert len(cat) == 1, "there should only be one foreshock"
    return cat[0]


@pytest.fixture(scope="session")
def crandall_inventory(crandall_ds):
    """ Return the inventory for the crandall dataset."""
    return crandall_ds.station_client.get_stations()


@pytest.fixture(scope="session")
def crandall_stream(crandall_event, crandall_ds, crandall_inventory):
    """ Return the streams from the crandall event, remove response """
    time = obsplus.get_reference_time(crandall_event)
    t1 = time - 5
    t2 = time + 60
    st = crandall_ds.waveform_client.get_waveforms(starttime=t1, endtime=t2)
    st.detrend("linear")
    prefilt = [0.1, 0.2, 40, 50]
    st.remove_response(crandall_inventory, pre_filt=prefilt, output="VEL")
    return st


@pytest.fixture(scope="session")
def crandall_st_dict_unified(crandall_ds, crandall_inventory):
    """
    Return a stream dict for the crandall dataset where each stream has been
    re-sampled to 100 Hz
    """
    fetcher = crandall_ds.get_fetcher()
    st_dict = dict(fetcher.get_event_waveforms(time_before=10, time_after=190))
    # re-sample all traces to 100 Hz
    for _, st in st_dict.items():
        st.resample(40)
        st.detrend("linear")
        prefilt = [0.1, 0.2, 40, 50]
        st.remove_response(crandall_inventory, pre_filt=prefilt, output="DISP")
    # remove response
    return st_dict


@pytest.fixture(scope="session")
def spectrum_group_crandall(request):
    """ Init a big source object on crandall data. """
    cache_path = Path(TEST_DATA_CACHE) / "crandall_source_group.pkl"
    if not cache_path.exists():
        crandall_st_dict_unified = request.getfixturevalue("crandall_st_dict_unified")
        crandall_catalog = request.getfixturevalue("crandall_catalog")
        crandall_inventory = request.getfixturevalue("crandall_inventory")
        sg = SpectrumGroup(
            crandall_st_dict_unified, crandall_catalog, crandall_inventory
        )
        sg.to_pickle(cache_path)
    else:
        sg = SpectrumGroup.from_pickle(cache_path)
    assert not hasattr(sg.stats, "process") or not sg.stats.processing
    assert not (sg.data == 0).all().all()
    return sg


# ------- Fixtures for Coal Node dataset


@pytest.fixture(scope="session")
def node_dataset():
    """ Return a dataset of the node data. """
    return obsplus.load_dataset("coal_node")


@pytest.fixture(scope="session")
def node_st(node_dataset):
    """ Return a stream containing data for the entire node dataset. """

    def remove_response(stt) -> obspy.Stream:
        """ using the fairfield files, remove the response through deconvolution """
        stt.detrend("linear")
        paz_5hz = corn_freq_2_paz(5.0, damp=0.707)
        paz_5hz["sensitivity"] = 76700
        pre_filt = (0.25, 0.5, 200.0, 250.0)
        stt.simulate(paz_remove=paz_5hz, pre_filt=pre_filt)
        return stt

    fetcher = node_dataset.get_fetcher()
    stream = obspy.Stream()
    for eid, st in fetcher.yield_event_waveforms(time_before=10, time_after=10):
        stream += remove_response(st)
    return stream


@pytest.fixture(scope="session")
def node_catalog(node_dataset):
    """ return the node catalog. """
    return node_dataset.event_client.get_events()


@pytest.fixture(scope="session")
def node_catalog_no_picks(node_catalog):
    """ return the node catalog with just origins """
    eid_map = {}
    cat = Catalog()
    for num, eve in enumerate(node_catalog):
        eve_out = Event(origins=eve.origins)
        for o in eve_out.origins:
            o.arrivals = []
        eve_out.resource_id = ResourceIdentifier(f"event_{num}")
        cat.append(eve_out)
        eid_map[eve.resource_id.id] = eve_out.resource_id.id
    return cat, eid_map


# @pytest.fixture(scope="session")
# def node_st_dict_no_picks(node_catalog_no_picks, node_st_dict):
#    st_dict = {}
#    for key in node_catalog_no_picks[1]:
#        st_dict[node_catalog_no_picks[1][key]] = node_st_dict[key]
#    return st_dict


@pytest.fixture(scope="session")
def node_inventory(node_dataset):
    """ get the inventory of the node dataset. """
    inv = node_dataset.station_client.get_stations()
    # Go through and make sure the azimuth and dip are populated
    for net in inv:
        for sta in net:
            for chan in sta:
                chan.azimuth = 0
                chan.dip = 0
    return inv


@pytest.fixture(scope="session")
def node_stats_group(node_st, node_catalog, node_inventory):
    """ Return a StatsGroup object from the node dataset. """
    # TODO: The arrivals on this catalog all seem to point to rejected picks,
    #  which seems a little unhelpful?
    kwargs = dict(catalog=node_catalog, inventory=node_inventory, restrict_to_arrivals=False)
    return mopy.core.statsgroup.StatsGroup(**kwargs)


@pytest.fixture(scope="session")
def node_stats_group_no_picks(node_catalog_no_picks, node_inventory):
    """ return a StatsGroup for a catalog that doesn't have any picks """
    # This will probably need to be refactored in the future, but for now...
    kwargs = dict(catalog=node_catalog_no_picks[0], inventory=node_inventory)
    return mopy.core.statsgroup.StatsGroup(**kwargs)


@pytest.fixture(scope="session")
def node_trace_group_raw(node_stats_group, node_st):
    """ Return a trace group from the node data. """
    return mopy.core.tracegroup.TraceGroup(
        node_stats_group, waveforms=node_st, motion_type="velocity"
    )


@pytest.fixture(scope="session")
def node_trace_group(node_trace_group_raw):
    """ Return a trace group from the node data. """
    # fill NaN with zero and return
    return node_trace_group_raw.fillna()


@pytest.fixture(scope="session")
def spectrum_group_node_session(node_trace_group):
    """ Return a source group with node data. """
    return node_trace_group.fft()
    # tg = node_trace_group
    # kwargs = dict(data=tg.data, channel_info=tg.channel_info, stats=tg.stats)
    # sg = mopy.core.spectrumgroup.SpectrumGroup(**kwargs)
    # assert not (sg.data == 0).all().all()
    # return sg


@pytest.fixture
def spectrum_group_node(spectrum_group_node_session):
    """ Get the source group, return copy for possible mutation. """
    return spectrum_group_node_session.copy()


# --- collect all source groups


@pytest.fixture
def spectrum_group(spectrum_group_crandall):
    """ Return a copy of the crandall spectrum group for testing """
    return spectrum_group_crandall.copy()
