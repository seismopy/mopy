"""
Tests for basics of SourceGroup and friends.
"""
from __future__ import annotations
import pytest
from typing import Callable, List

import matplotlib.pyplot as plt  # Imported for ease of debugging
import numpy as np
from numpy.testing import assert_allclose as np_assert
import pandas as pd
from scipy.fftpack import next_fast_len

from obspy import Stream, Trace
from obsplus.constants import NSLC
from obsplus.utils.time import to_utc

import mopy
from mopy import SpectrumGroup, StatsGroup, TraceGroup
from mopy.testing import gauss, gauss_deriv, gauss_deriv_deriv


# --- Fixtures

# Constants for gauss wave
_t1, _t2, _dt = 0, 10.24, 0.01  # Need to be very particular here to avoid any zero padding
_a, _b, _c = 0.1, 5, np.sqrt(3)
_t = np.arange(_t1, _t2+1, _dt)


@pytest.fixture(scope="function")
def gauss_stat_group(node_stats_group) -> StatsGroup:
    """
    Create a StatsGroup to feed to the TraceGroup for FFT calcs
    """
    # Two streams, one has the full data, the other has only part of it
    # (to see the effects of zero padding the data)
    df_contents = {
        "event_id": ["event1", "event1"],
        "seed_id": ["UK.STA1..HHZ", "UK.STA2..HHZ"],
        "seed_id_less": ["UK.STA1..HH", "UK.STA2..HH"],
        "phase_hint": ["P", "P"],
        "time": [np.datetime64("2020-01-01T00:00:01"), np.datetime64("2020-01-01T00:00:01")],
        "starttime": [np.datetime64("2020-01-01T00:00:00"), np.datetime64("2020-01-01T00:00:00"),],
        "endtime": [
            np.datetime64("2020-01-01T00:00:00") + np.timedelta64(int(_t2*1000), 'ms') - np.timedelta64(int(_dt*1000), 'ms'),
            np.datetime64("2020-01-01T00:00:00") + np.timedelta64(int(_t2//2*1000), 'ms') - np.timedelta64(int(_dt*1000), 'ms'),
        ],
        "ray_path_length_m": [2000, 2000],
        "hyp_distance_m": [2000, 2000],
    }
    df = pd.DataFrame(df_contents, columns=df_contents.keys())
    df["sampling_rate"] = 1 / _dt
    df.set_index(['phase_hint', 'event_id', 'seed_id_less', 'seed_id'], inplace=True)
    return node_stats_group.new_from_dict(data=df)  # This is a weird way to do this


@pytest.fixture(scope="function")
def gauss_trace_group(gauss_stat_group) -> TraceGroup:
    """ Create a TraceGroup with a Gaussian pulse as the data """
    # Generate the data
    data = gauss(_t, _a, _b, _c)
    gauss_stat_group.data["sampling_rate"] = 1/_dt
    # Build a stream from the data
    tr = Trace(data=data, header={
        "starttime": to_utc(gauss_stat_group.data.iloc[0].starttime),
        "delta": _dt,
        "network": "UK",
        "station": "STA1",
        "channel": "HHZ",
    })
    st = Stream()
    st.append(tr)
    # Add a second trace with a substantial discontinuity caused by zero-padding
    st.append(tr.copy())  # Same data, but the time window in the StatsGroup halves it
    st[1].stats.station = "STA2"
    # Make a TraceGroup
    return mopy.TraceGroup(gauss_stat_group, st, "displacement").fillna()


@pytest.fixture(scope="function")
def gauss_spec_group(gauss_trace_group) -> SpectrumGroup:
    """ Make a SpectrumGroup by calculating the DFT """
    return gauss_trace_group.dft()


# --- Tests

class TestSpectrumGroupBasics:
    """ Tests for basics for source group. """

    def test_instance(self, spectrum_group_node):
        assert isinstance(spectrum_group_node, mopy.SpectrumGroup)
        assert isinstance(spectrum_group_node.data, pd.DataFrame)

    def test_processing(self, spectrum_group_node):
        """ Ensure there is no processing until performing an operation. """
        # the NaNs should have been filled, hence it shows up in processing
        assert ["fillna" in x for x in spectrum_group_node.processing]
        # next create a new spectrum group, ensure processing increases
        start_len = len(spectrum_group_node.processing)
        out = abs(spectrum_group_node)
        assert ["abs" in x for x in out.processing]
        assert start_len < len(out.processing)
        # the original should be unchanged
        assert len(spectrum_group_node.processing) == start_len

    def test_pickle(self, spectrum_group_node):
        """ ensure the source group can be pickled and read """
        byts = spectrum_group_node.to_pickle()
        sg = spectrum_group_node.from_pickle(byts)
        assert sg.data.equals(spectrum_group_node.data)

    def test_labels(self, spectrum_group_node):
        """ ensure the columns have the appropriate label. """
        df = spectrum_group_node.data
        col_name = df.columns.name
        assert col_name == "frequency"


class TestExpandCollapseSeedId:
    """ tests for expanding and collapsing seed ids. """

    def test_expand_seed_id(self, spectrum_group_node):
        """ tests for expanding index to include seed codes. """
        sg = spectrum_group_node.expand_seed_id()
        inds = sg.data.index
        assert "seed_id" not in inds.names
        assert set(inds.names).issuperset(NSLC)
        # ensure calling a second time does nothing

    def test_collapse_seed_id(self, spectrum_group_node):
        """ ensure the seed_ids can be collapsed back down. """
        # ind_origin = source_group_node.data.index
        # sg = source_group_node.expand_seed_id().collapse_seed_id()
        # assert (sg.data.index == ind_origin).all()


class TestProcessingProperties:
    """ Tests for properties that reveal if certain processes have taken place. """

    def test_spreading(self, spectrum_group_node):
        """ test spreading """
        # spreading has not been corrected
        assert not spectrum_group_node.spreading_corrected
        out = spectrum_group_node.correct_spreading()
        assert out.spreading_corrected

    def test_distance(self, spectrum_group_node):
        """ tests for attenuation. """
        # attenuation has not yet been corrected
        assert not spectrum_group_node.attenuation_corrected
        out = spectrum_group_node.correct_attenuation(100)
        assert out.attenuation_corrected

    def test_radiation_pattern(self, spectrum_group_node):
        """ tests source radiation pattern  correction. """
        assert not spectrum_group_node.radiation_pattern_corrected
        out = spectrum_group_node.correct_radiation_pattern()
        assert out.radiation_pattern_corrected


class TestSpectrumGroupOperations:
    """ Tests for operations on the SourceGroup. """

    @pytest.fixture(scope="class")
    def smoothed_group(self, spectrum_group_node) -> SpectrumGroup:
        """ return a smoothed group using default params. """
        return spectrum_group_node.ko_smooth()

    def test_ko_smooth_no_resample(self, spectrum_group_node):
        """ Ensure the smoothing was applied. """
        smooth_no_resample = spectrum_group_node.ko_smooth()
        assert isinstance(smooth_no_resample, mopy.SpectrumGroup)

    def test_ko_smooth_refactor(self, spectrum_group_node):
        """ Ensure the source group can be sampled with smoothing. """
        # get log space frequencies
        sampling_rate = spectrum_group_node.sampling_rate / 2.0
        freqs = np.logspace(0, np.log10(sampling_rate), 22)[1:-1]
        smooth = spectrum_group_node.ko_smooth(frequencies=freqs)
        assert np.all(smooth.data.columns.values == freqs)

    def test_log_resample(self, spectrum_group_node):
        """ Ensure the resampling works by testing it resamples
            to the given length. """
        # log-resample to half the number of frequency samples
        length = int(len(spectrum_group_node.data.columns) / 2)

        resampd = spectrum_group_node.log_resample_spectra(length)

        assert np.all(len(resampd.data.columns) == length)

    def test_break_log_resample(self, spectrum_group_node):
        """ Ensure the resampling throws an error when the
             length of resampling > current number of
             samples. """
        length = len(spectrum_group_node.data.columns) + 1
        with pytest.raises(ValueError):
            spectrum_group_node.log_resample_spectra(length)

    def test_normalized(self, spectrum_group_node):
        """ ensure the normalization for """
        sg = abs(spectrum_group_node)
        norm = sg.normalize()
        assert (sg.data >= norm.data).all().all()

    def test_subtract_noise(self, spectrum_group_node):
        """ Ensure subtracting a phase works. """
        sg = abs(spectrum_group_node).normalize()  # take abs to avoid complex
        phase_hint = "Noise"
        # now subtract noise
        nsg = sg.subtract_phase(phase_hint=phase_hint, drop=False)
        assert isinstance(nsg, mopy.SpectrumGroup)
        # ensure all values are less than or equal
        for phase in sg.data.index.get_level_values("phase_hint").unique():
            df_pre_sub = sg.data.loc[phase]
            df_post_sub = nsg.data.loc[phase]
            con1 = df_post_sub <= df_pre_sub
            con2 = df_post_sub.isnull()
            assert (con1 | con2).all().all()

    def test_mask_signal_below_noise(self, spectrum_group_node):
        """ Ensure the signal below the noise can be masked. """
        sg = spectrum_group_node.abs().normalize()
        phase_hint = "Noise"
        nsg = sg.mask_by_phase(phase_hint=phase_hint, drop=False)
        df = nsg.data
        # All the noise entries should have been masked
        assert df.loc["Noise"].isnull().all().all()

    def test_correct_geometric_spreading(self, spectrum_group_node):
        """ ensure the geometric spreading bit works. """
        sg = spectrum_group_node.abs().correct_spreading()
        assert (spectrum_group_node.data <= sg.data).all().all()

    def test_correct_radiation_pattern(self, spectrum_group_node):
        """ Tests for correcting geometric spreading. """
        sg = spectrum_group_node.abs().correct_radiation_pattern()
        df1, df2 = spectrum_group_node.data, sg.data
        # ensure noise was dropped
        assert not df2.isnull().any().any()
        assert (df1.loc[df2.index] <= df2).all().all()


class TestSpectraConversions:
    """ Tests for converting back and forth between discrete and continuous transforms """

    @pytest.fixture
    def time_domain_cft_equivalent(self, gauss_trace_group) -> List:
        data = gauss_trace_group.data
        stats = gauss_trace_group.stats
        data_int = data.cumsum(axis=1)
        return [(data_int.iloc[i, stats.iloc[i].npts - 1] - data_int.iloc[i, 0]) / stats.iloc[i].sampling_rate for i in
                range(len(data_int))]

    @pytest.fixture
    def time_domain_psd_equivalent(self, gauss_trace_group) -> List:
        data = gauss_trace_group.data
        stats = gauss_trace_group.stats
        data_sq_int = (data ** 2).cumsum(axis=1)
        return [(data_sq_int.iloc[i, stats.iloc[i].npts - 1] - data_sq_int.iloc[i, 0]) / stats.iloc[i].sampling_rate for i in
                range(len(data_sq_int))]

    @pytest.fixture
    def dft(self, gauss_spec_group) -> SpectrumGroup:
        return gauss_spec_group

    @pytest.fixture
    def cft(self, gauss_spec_group) -> SpectrumGroup:
        return gauss_spec_group.to_spectra_type("cft")

    @pytest.fixture
    def psd(self, gauss_spec_group) -> SpectrumGroup:
        return gauss_spec_group.to_spectra_type("psd")

    def test_dft_to_dft(self, dft):
        """ Verify that going to itself doesn't change the data"""
        df = dft.data.copy()
        conv = dft.to_spectra_type("dft")
        assert df.equals(conv.data)

    def test_dft_to_cft(self, gauss_spec_group, time_domain_cft_equivalent):
        # Repeat in the frequency domain
        conv = gauss_spec_group.to_spectra_type('cft')
        fd = conv.abs().data.max(axis=1)
        np_assert(time_domain_cft_equivalent, fd, rtol=1e-4)

    def test_cft_to_dft(self, cft, dft):
        """ Because if it works one way, we might as well support the reverse """
        conv = cft.to_spectra_type("dft")
        np_assert(conv.data, dft.data)

    def test_dft_to_psd(self, dft, time_domain_psd_equivalent):
        conv = dft.to_spectra_type("psd")
        fd = np.sum(conv.abs().data, axis=1)
        assert len(fd) == 2
        np_assert(fd[0], time_domain_psd_equivalent[0], rtol=1e-6)
        np_assert(fd[1], time_domain_psd_equivalent[1], atol=0.02)

    def test_psd_to_dft(self, psd, dft):
        with pytest.warns(UserWarning, match="loss of sign information"):
            conv = psd.to_spectra_type("dft")
        np_assert(conv.abs().data, dft.abs().data)

    def test_cft_to_psd(self, cft, psd):
        conv = cft.to_spectra_type("psd")
        np_assert(conv.data, psd.data)

    def test_psd_to_cft(self, psd, cft):
        with pytest.warns(UserWarning, match="loss of sign information"):
            conv = psd.to_spectra_type("cft")
        np_assert(conv.abs().data, cft.abs().data)

    def test_invalid_spectra_type(self, dft):
        with pytest.raises(ValueError, match="Invalid spectra type"):
            dft.to_spectra_type("rainbow")

    def test_spectra_motion_type_conversion(self, dft):
        check = dft.to_motion_type("velocity").to_spectra_type("cft")
        conv = dft.to_spectra_type("cft", motion_type="velocity")
        np_assert(conv.data, check.data)

    def test_spectra_with_smoothing(self, dft):
        check = dft.apply_smoothing("ko_smooth").to_spectra_type("cft")
        conv = dft.to_spectra_type("cft", smoothing="ko_smooth")
        np_assert(conv.data, check.data)


class TestApplySmoothing:
    def test_apply_ko_smoothing(self, spectrum_group_node):
        frequencies = spectrum_group_node.data.columns.values[::3]
        smoothed = spectrum_group_node.apply_smoothing("ko_smooth", frequencies=frequencies)
        assert round(len(spectrum_group_node.data.columns)/len(smoothed.data.columns)) == 3

    def test_invalid_smoothing(self, spectrum_group_node):
        with pytest.raises(ValueError, match="Invalid smoothing"):
            spectrum_group_node.apply_smoothing("silky")


class TestSpectralSource:
    """ Tests for calculating source params directly from spectra. """

    # Fixtures

    @pytest.fixture
    def fft(self, node_trace_group) -> SpectrumGroup:
        """ Calculate spectra using numpy """
        return node_trace_group.dft()

    @pytest.fixture
    def mtspec1(self, node_trace_group) -> SpectrumGroup:
        """ Calculate spectra using mtspec """
        pytest.importorskip("mtspec")
        return node_trace_group.mtspec(to_dft=True)

    @pytest.fixture(scope="function", params=("fft", "mtspec1"))
    def spec_group_for_source_params(self, request) -> SpectrumGroup:
        """ Gather different methods for calculating spectra """
        return request.getfixturevalue(request.param)

    @pytest.fixture(scope="function")
    def source_params(self, spec_group_for_source_params) -> pd.DataFrame:
        """ Return a df of calculated source info from the SpectrumGroup """
        # Apply pre-processing
        out = spec_group_for_source_params.abs().ko_smooth()
        # out = spec_group_for_source_params.abs().ko_smooth()
        out = out.correct_radiation_pattern()
        out = out.correct_spreading()
        out = out.correct_attenuation()
        out = out.correct_free_surface()
        # Return source df
        return out.calc_source_params()

    @pytest.fixture(scope="function")
    def source_params_preprocessing(self, spec_group_for_source_params) -> pd.DataFrame:
        """
        Return a df of calculated source info from the SpectrumGroup where
        preprocessing occurred in the calculation call
        """
        return spec_group_for_source_params.calc_source_params(apply_corrections=True, smoothing="ko_smooth")

    @pytest.fixture(scope="function")
    def source_params_no_preprocessing(self, spec_group_for_source_params) -> pd.DataFrame:
        """ Calculate source params without doing any kind of preprocessing """
        with pytest.warns(UserWarning, match="has not been corrected"):
            return spec_group_for_source_params.calc_source_params()

    # Tests

    def test_basic(self, source_params):
        """ Ensure output is a nonempty DataFrame """
        assert isinstance(source_params, pd.DataFrame)
        assert not source_params.empty

    def test_preprocessing(self, source_params, source_params_preprocessing):
        """ Make sure that preprocessing was applied when desired """
        np_assert(source_params_preprocessing, source_params, rtol=0.05)

    def test_no_preprocessing(self, spectrum_group_node, source_params):
        """ Make sure that preprocessing is not applied when not desired """
        with pytest.warns(UserWarning, match="has not been corrected"):
            df = spectrum_group_node.calc_source_params()
        assert not np.allclose(df, source_params, rtol=0.05)

    def test_values(self, source_params):
        """ Ensure values are plausible """
        # This is intentionally somewhat brittle and intended to catch
        # subtle yet significant changes to the parameter calculations
        medians = source_params.median()
        checks = {
            "fc": (5.8, 0.02),  # (value, rtol)
            "omega0": (6.35e-5, 0.12),
            "moment": (1.3e11, 0.12),
            "potency": (3.05, 0.12),
            "energy": (1.17e7, 0.02),
            "mw": (1.34, 0.03)}
        for key, (val, tol) in checks.items():
            np_assert(medians[key], val, rtol=tol)


class TestGroundMotionConversions:  # TODO: These still aren't matching as close as I would like, particularly at higher frequencies
    """ tests for converting from one ground motion type to another. """

    # Helper functions

    def check_ground_motion(self, calculated: SpectrumGroup, theoretical: list, motion_type: str, rtol1: float = 0.1, rtol2: float = 1) -> None:
        """ Evaluate the result of a ground motion conversion """
        data = calculated.data
        # Make sure the motion type was updated
        assert calculated.motion_type == motion_type
        # Make sure the data for each trace match the theoretical data
        np_assert(abs(data.iloc[0]).mean(), abs(theoretical[0]).mean(), rtol=rtol1)
        np_assert(abs(data.iloc[1]).max(), abs(theoretical[1]).max(), rtol=rtol2)  # This is so far off that it kind of makes me wonder 'why bother'...

    def build_spectra(self, func: Callable, fft_len: int):
        """ Build the fft spectra for the given motion type """
        # Create the spectra for the full-length displacement data
        # full = np.zeros(fft_len)
        # full[0:len(_t)] = func(_t, _a, _b, _c)
        full = func(_t, _a, _b, _c)
        full = np.fft.rfft(full)  # , n=fft_len)
        # Create the spectra for the half-length (zero padded) displacement data
        half = np.zeros(fft_len)
        half[0:len(_t)//2] = func(_t[0:len(_t)//2], _a, _b, _c)
        half = np.fft.rfft(half, n=fft_len)
        return full, half

    # Fixtures

    @pytest.fixture(scope="function")
    def displacement_spec_group(self, gauss_spec_group) -> SpectrumGroup:
        """ Return a SpectrumGroup with displacement ground motion """
        return gauss_spec_group

    @pytest.fixture(scope="function")
    def velocity_spec_group(self, gauss_spec_group) -> SpectrumGroup:
        """ Return a SpectrumGroup with velocity ground motion """
        return gauss_spec_group.to_motion_type("velocity")

    @pytest.fixture(scope="function")
    def acceleration_spec_group(self, gauss_spec_group) -> SpectrumGroup:
        """ Return a SpectrumGroup with acceleration ground motion """
        return gauss_spec_group.to_motion_type("acceleration")

    @pytest.fixture(scope="class")
    def fast_len(self) -> int:
        """ Determine the appropriate length of the fft data array """
        return next_fast_len(len(_t) + 1)

    @pytest.fixture(scope="class")
    def displacement_spectra(self, fast_len) -> SpectrumGroup:
        """ Return the analytically calculated displacement spectra """
        return self.build_spectra(gauss, fast_len)

    @pytest.fixture(scope="class")
    def velocity_spectra(self, fast_len) -> SpectrumGroup:
        """ Return the analytically calculated velocity spectra """
        return self.build_spectra(gauss_deriv, fast_len)

    @pytest.fixture(scope="class")
    def acceleration_spectra(self, fast_len) -> SpectrumGroup:
        """ Return the analytically calculated acceleration spectra """
        return self.build_spectra(gauss_deriv_deriv, fast_len)

    # Tests

    def test_displacement_to_velocity(self, displacement_spec_group, velocity_spectra, displacement_spectra):
        """ Test for converting from displacement to velocity """
        mt = "velocity"
        self.check_ground_motion(displacement_spec_group.to_motion_type(mt), velocity_spectra, mt)

    def test_velocity_to_displacement(self, velocity_spec_group, displacement_spectra):
        """ Test for converting from velocity to displacement. """
        mt = "displacement"
        self.check_ground_motion(velocity_spec_group.to_motion_type(mt), displacement_spectra, mt, rtol1=0.6)

    def test_displacement_to_acceleration(self, displacement_spec_group, acceleration_spectra):
        """ Test for converting from displacement to acceleration """
        mt = "acceleration"
        self.check_ground_motion(displacement_spec_group.to_motion_type(mt), acceleration_spectra, mt, rtol2=120)  # TODO: The difference on this concerns me a bit

    def test_acceleration_to_displacement(self, acceleration_spec_group, displacement_spectra):
        """ Test for converting from acceleration to displacement """
        mt = "displacement"
        self.check_ground_motion(acceleration_spec_group.to_motion_type(mt), displacement_spectra, mt, rtol1=0.6)

    def test_velocity_to_acceleration(self, velocity_spec_group, acceleration_spectra):
        """ Test for converting from velocity to acceleration """
        mt = "acceleration"
        self.check_ground_motion(velocity_spec_group.to_motion_type(mt), acceleration_spectra, mt, rtol2=120)  # TODO: Same with this one

    def test_acceleration_to_velocity(self, acceleration_spec_group, velocity_spectra):
        """ Test for converting from acceleration to velocity """
        mt = "velocity"
        self.check_ground_motion(acceleration_spec_group.to_motion_type(mt), velocity_spectra, mt, rtol1=0.2)

    def test_bad_value_raises(self, spectrum_group_node):
        """ ensure a non-supported motion type raises an error. """
        with pytest.raises(ValueError):
            spectrum_group_node.to_motion_type("super_velocity")

    def test_velocity_to_velocity(self, spectrum_group_node):
        """ Ensure converting velocity to velocity works. """
        sg = spectrum_group_node.to_motion_type("velocity")
        assert isinstance(sg, mopy.SpectrumGroup)
        assert spectrum_group_node.motion_type == "velocity"
        (spectrum_group_node.data == sg.data).all().all()


class TestCorrectQualityFactor:
    """ Ensure the quality factor can be corrected for. """

    def test_scalar(self, spectrum_group_node):
        """ ensure the quality factor can be used as an int. """
        sg = spectrum_group_node.abs().ko_smooth()
        out = sg.correct_attenuation(drop=True)
        # since drop == True the noise should have been dropped
        phase_hints = np.unique(out.data.index.get_level_values("phase_hint"))
        assert "Noise" not in phase_hints
        # make sure non-zero values have all increased or stayed the same
        for phase in phase_hints:
            df1, df2 = sg.data.loc[phase], out.data.loc[phase]
            assert (df1 <= df2).all().all()
