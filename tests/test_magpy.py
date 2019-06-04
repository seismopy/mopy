"""
Tests to compare with magpy
"""
from __future__ import annotations


# import magpy
# import mopy
# from mopy.config import get_default_param
#
# import obsplus
# from magpy import CalcMl, CalcMw, Spectra
#
# import pytest
#
#
# @pytest.fixture(scope="session")
# def mag_calc_mw(node_dataset_old, node_st_dict):
#     """ """
#     # first get obspy stuff from node_dataset_old
#     inv = node_dataset_old.station_client.get_stations()
#     cat = node_dataset_old.event_client.get_events()
#     eve = cat[0]
#
#     st = node_st_dict[str(eve.resource_id)].integrate().detrend("linear")
#
#     p_params = {
#         "q_factor": get_default_param("quality_factor"),
#         # Optional, default is 1000... ideally would somehow like to estimate this on the fly
#         "radiation_pattern": get_default_param("p_radiation_coefficient"),  # Required
#     }
#     s_params = {
#         "q_factor": get_default_param("quality_factor"),
#         "radiation_pattern": get_default_param("p_radiation_coefficient"),
#     }
#     mag_specific_params = {
#         "use_p": True,  # Use P-waves in magnitude calculations (True by default)
#         "use_s": True,  # Use S-waves in magnitude calculations (True by default)
#         "vp": get_default_param("p_velocity"),  # Required
#         "vs": get_default_param("s_velocity"),  # Required
#         "density": get_default_param("density"),  # Required
#         "p_params": p_params,  # Required if use_p is True
#         "s_params": s_params,  # Required if use_s is True
#         "method": "np_fft_savgol",  # Method to use to calculate the displacement spectra
#     }
#
#     waveform_params = dict(time_before=5, time_after=5)
#     cmw = CalcMw(
#         inv,
#         exclude=["EL*", "EN*", "HL*", "HN*"],
#         mag_specific_params=mag_specific_params,
#         set_preferred=True,
#         status="preliminary",
#         time_before=5,
#         remove_response=False,
#         time_after=5,
#     )  # The actual time window that is used is much shorter than 10 seconds, but it just needs to know how much data to pull to start
#
#     breakpoint()
#     cmw.event_mag(eve, waveforms=st)
#     eve.preferred_magnitude()
#
#
# def test_sumfin(mag_calc_mw):
#     """ tests sumfin"""
