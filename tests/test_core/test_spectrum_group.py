"""
Tests for basics of SourceGroup and friends.
"""
import numpy as np
import pandas as pd
import pytest
from obsplus.constants import NSLC

from mopy.core import SpectrumGroup


class TestSpectrumGroupBasics:
    """ Tests for basics for source group. """

    def test_instance(self, source_group_node):
        assert isinstance(source_group_node, SpectrumGroup)
        assert isinstance(source_group_node.data, pd.DataFrame)

    def test_processing(self, source_group_node):
        """ Ensure there is no processing until performing an operation. """
        assert not getattr(source_group_node.stats, "processing", None)
        out = abs(source_group_node)
        assert hasattr(out.stats, "processing")
        has_absolute = ["abs" in x for x in out.stats.processing]
        assert any(has_absolute)

    def test_pickle(self, source_group_node):
        """ ensure the source group can be pickled and read """
        byts = source_group_node.to_pickle()
        sg = source_group_node.from_pickle(byts)
        assert sg.data.equals(source_group_node.data)

    def test_labels(self, source_group_node):
        """ ensure the columns have the appropriate label. """
        df = source_group_node.data
        col_name = df.columns.name
        assert col_name == 'frequency'


class TestExpandCollapseSeedId:
    """ tests for expanding and collapsing seed ids. """

    def test_expand_seed_id(self, source_group_node):
        """ tests for expanding index to include seed codes. """
        sg = source_group_node.expand_seed_id()
        inds = sg.data.index
        assert 'seed_id' not in inds.names
        assert set(inds.names).issuperset(NSLC)
        # ensure calling a second time does nothing

    def test_collapse_seed_id(self, source_group_node):
        """ ensure the seed_ids can be collapsed back down. """
        # ind_origin = source_group_node.data.index
        # sg = source_group_node.expand_seed_id().collapse_seed_id()
        # assert (sg.data.index == ind_origin).all()


class TestProcessingProperties:
    """ Tests for properties that reveal if certain processes have taken place. """

    def test_spreading(self, source_group_node):
        """ test spreading """
        # spreading has not been corrected
        assert not source_group_node.spreading_corrected
        out = source_group_node.correct_spreading()
        assert out.spreading_corrected

    def test_distance(self, source_group_node):
        """ tests for attenuation. """
        # attenuation has not yet been corrected
        assert not source_group_node.attenuation_corrected
        out = source_group_node.correct_attenuation(100)
        assert out.attenuation_corrected

    def test_radiation_pattern(self, source_group_node):
        """ tests source radiation pattern  correction. """
        assert not source_group_node.radiation_pattern_corrected
        out = source_group_node.correct_radiation_pattern()
        assert out.radiation_pattern_corrected


class TestMeta:
    """ tests for the meta dict holding info about the source_group. """

    def test_phase_window_times(self, source_group_node):
        meta = source_group_node.meta
        start = meta["tw_start"]
        pick = meta["time"]
        end = meta["tw_end"]
        # because a buffer is added pick should be greater than start
        assert (pick > start).all()
        assert (pick < end).all()
        assert (end > start).all()


class TestSourceGroupOperations:
    """ Tests for operations on the SourceGroup. """

    @pytest.fixture(scope="class")
    def smoothed_group(self, source_group_node):
        """ return a smoothed group using default params. """
        return source_group_node.ko_smooth()

    def test_ko_smooth_no_resample(self, source_group_node):
        """ Ensure the smoothing was applied. """
        smooth_no_resample = source_group_node.ko_smooth()
        assert isinstance(smooth_no_resample, SpectrumGroup)

    def test_ko_smooth_refactor(self, source_group_node):
        """ Ensure the source group can be sampled with smoothing. """
        # get log space frequencies
        sampling_rate = source_group_node.stats.sampling_rate / 2.0
        freqs = np.logspace(0, np.log10(sampling_rate), 22)[1:-1]
        smooth = source_group_node.ko_smooth(frequencies=freqs)
        assert np.all(smooth.data.columns.values == freqs)

    def test_normalized(self, source_group_node):
        """ ensure the normalization for """
        sg = abs(source_group_node)
        norm = sg.normalize()
        assert (sg.data >= norm.data).all().all()

    def test_subtract_noise(self, source_group_node):
        """ Ensure subtracting a phase works. """
        sg = abs(source_group_node).normalize()  # take abs to avoid complex
        phase_hint = "Noise"
        # now subtract noise
        nsg = sg.subtract_phase(phase_hint=phase_hint, drop=False)
        assert isinstance(nsg, SpectrumGroup)
        # ensure all values are less than or equal
        for phase in sg.data.index.get_level_values("phase_hint"):
            df_pre_sub = sg.data.loc[phase]
            df_post_sub = nsg.data.loc[phase]
            con1 = df_post_sub <= df_pre_sub
            con2 = df_post_sub.isnull()
            assert (con1 | con2).all().all()

    def test_mask_signal_below_noise(self, source_group_node):
        """ Ensure the signal below the noise can be masked. """
        sg = source_group_node.abs().normalize()
        phase_hint = "Noise"
        nsg = sg.mask_by_phase(phase_hint=phase_hint, drop=False)
        df = nsg.data
        # All the noise entries should have been masked
        assert df.loc["Noise"].isnull().all().all()

    def test_correct_geometric_spreading(self, source_group_node):
        """ ensure the geometric spreading bit works. """
        sg = source_group_node.abs().correct_spreading()
        assert (source_group_node.data <= sg.data).all().all()

    def test_correct_radiation_pattern(self, source_group_node):
        """ Tests for correcting geometric spreading. """
        sg = source_group_node.abs().correct_radiation_pattern()
        df1, df2 = source_group_node.data, sg.data
        # ensure noise was dropped
        assert not df2.isnull().any().any()
        assert (df1.loc[df2.index] <= df2).all().all()


class TestSpectralSource:
    """ Tests for calculating spectral characteristics directly from spectra. """

    def test_basic(self, source_group_node):
        """ Ensure mopy is returned. """
        # apply pre-processing
        out = source_group_node.abs().ko_smooth().correct_radiation_pattern()
        out = out.correct_spreading() #.correct_attenuation()
        out = out.correct_free_surface()
        # calculate simple source_df
        out = out.calc_simple_source()
        source_df = out.source_df

        assert isinstance(source_df, pd.DataFrame)
        assert not source_df.empty


class TestMagnitudes:
    """ """


class TestGroundMotionConversions:
    """ tests for converting from one ground motion type to another. """

    def test_velocity_to_displacement(self, source_group_node):
        """ Tests for converting from velocity to displacement. """
        out = source_group_node.to_motion_type("displacement")
        assert out.stats.motion_type == "displacement"
        # make sure the data has changed
        assert (out.data != source_group_node.data).all().all()

    def test_bad_value_raises(self, source_group_node):
        """ ensure a non-supported motion type raises an error. """
        with pytest.raises(ValueError):
            source_group_node.to_motion_type("super_velocity")

    def test_velocity_to_acceleration(self, source_group_node):
        """ convert velocity to acceleration. """
        out = source_group_node.to_motion_type("acceleration")
        assert out.stats.motion_type == "acceleration"
        assert (out.data != source_group_node.data).all().all()

    def test_velocity_to_velocity(self, source_group_node):
        """ Ensure converting velocity to velocity works. """
        sg = source_group_node.to_motion_type("velocity")
        assert isinstance(sg, SpectrumGroup)
        (source_group_node.data == sg.data).all().all()


class TestCorrectQualityFactor:
    """ Ensure the quality factor can be corrected for. """

    def test_scalar(self, source_group_node):
        """ ensure the quality factor can be used as an int. """
        sg = source_group_node.abs().ko_smooth()
        out = sg.correct_attenuation(drop=True)
        # since drop == True the noise should have been dropped
        phase_hints = np.unique(out.data.index.get_level_values("phase_hint"))
        assert "Noise" not in phase_hints
        # make sure non-zero values have all increased or stayed the same
        for phase in phase_hints:
            df1, df2 = sg.data.loc[phase], out.data.loc[phase]
            assert (df1 <= df2).all().all()
