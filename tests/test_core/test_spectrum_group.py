"""
Tests for basics of SourceGroup and friends.
"""
from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose as np_assert
import pandas as pd
import pytest
from obsplus.constants import NSLC

from mopy import SpectrumGroup


class TestSpectrumGroupBasics:
    """ Tests for basics for source group. """

    def test_instance(self, spectrum_group_node):
        assert isinstance(spectrum_group_node, SpectrumGroup)
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


class TestSourceGroupOperations:
    """ Tests for operations on the SourceGroup. """

    @pytest.fixture(scope="class")
    def smoothed_group(self, spectrum_group_node):
        """ return a smoothed group using default params. """
        return spectrum_group_node.ko_smooth()

    def test_ko_smooth_no_resample(self, spectrum_group_node):
        """ Ensure the smoothing was applied. """
        smooth_no_resample = spectrum_group_node.ko_smooth()
        assert isinstance(smooth_no_resample, SpectrumGroup)

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
            resampd = spectrum_group_node.log_resample_spectra(length)

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
        assert isinstance(nsg, SpectrumGroup)
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


class TestSpectralSource:
    """ Tests for calculating source params directly from spectra. """

    @pytest.fixture(scope="function")
    def source_params(self, spectrum_group_node):
        """ Return a df of calculated source info from the SpectrumGroup """
        # Apply pre-processing
        out = spectrum_group_node.abs().ko_smooth()
        out = out.correct_radiation_pattern()
        out = out.correct_spreading()
        out = out.correct_attenuation()
        out = out.correct_free_surface()
        # Return source df
        return out.calc_source_params()

    def test_basic(self, source_params):
        """ Ensure output is a nonempty DataFrame """
        assert isinstance(source_params, pd.DataFrame)
        assert not source_params.empty

    def test_values(self, source_params):
        """ Ensure values are plausible """
        # This is intentionally somewhat brittle and intended to catch
        # subtle yet significant changes to the parameter calculations
        medians = source_params.median()
        checks = {
            "fc": 5.7,
            "omega0": 2.9e-3,
            "velocity_squared_integral": 2.5e-2,
            "moment": 5.8e12,
            "potency": 1.4e2,
            "energy": 1.0e7,
            "mw": 2.5}
        for key, val in checks.items():
            np_assert(medians[key], val, rtol=0.02)


class TestGroundMotionConversions:
    """ tests for converting from one ground motion type to another. """

    def test_velocity_to_displacement(self, spectrum_group_node):
        """ Tests for converting from velocity to displacement. """
        out = spectrum_group_node.to_motion_type("displacement")
        assert out.motion_type == "displacement"
        # make sure the data has changed
        assert (out.data != spectrum_group_node.data).all().all()

    def test_bad_value_raises(self, spectrum_group_node):
        """ ensure a non-supported motion type raises an error. """
        with pytest.raises(ValueError):
            spectrum_group_node.to_motion_type("super_velocity")

    def test_velocity_to_acceleration(self, spectrum_group_node):
        """ convert velocity to acceleration. """
        out = spectrum_group_node.to_motion_type("acceleration")
        assert out.motion_type == "acceleration"
        assert (out.data != spectrum_group_node.data).all().all()

    def test_velocity_to_velocity(self, spectrum_group_node):
        """ Ensure converting velocity to velocity works. """
        sg = spectrum_group_node.to_motion_type("velocity")
        assert isinstance(sg, SpectrumGroup)
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
