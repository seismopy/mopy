"""
Class for freq. domain data in mopy.
"""
from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.interpolate as interp

import mopy
from mopy.constants import _INDEX_NAMES, MOTION_TYPES, MOPY_SPECIFIC_PARAMS
from mopy.core.base import DataGroupBase
from mopy.smooth import konno_ohmachi_smoothing as ko_smooth
from mopy.sourcemodels import fit_model
from mopy.utils import _track_method


class SpectrumGroup(DataGroupBase):
    """
    A class to encompass many catalog sources.
    """

    min_samples = 60  # required number of samples per phase
    _default_velocity = {"P": 4000, "S": 2400}
    _default_radiation = {"P": 0.44, "S": 0.6}
    _default_quality_factor = 250
    _pre_taper = 0.10  # percentage of window added before/after phase to taper
    _max_taper_percentage = 0.05
    # The number of seconds per source-receiver distance to require for
    # phase windows, while still requiring a min number of samples.
    _seconds_per_m = 0.00003
    # the time domain data, useful for debugging
    _td_data = None
    # DF for storing info about source parameters
    source_df = None
    _log_resampled = False

    def __init__(self, data: pd.DataFrame, stats_group: mopy.StatsGroup):
        super().__init__(stats_group)
        if not set(self.stats.columns).issuperset(MOPY_SPECIFIC_PARAMS):
            warnings.warn(
                "MoPy specific parameters have not been applied to the StatsGroup. Applying default parameters"
            )
            self.stats_group.apply_defaults(inplace=True)
        # set stats
        self.data = data.copy()
        # init empty source dataframe
        self.source_df = pd.DataFrame(index=self.stats.index)

    # --- SpectrumGroup hooks

    def post_source_function_hook(self):
        """ A hook that gets called after each source function. """
        pass

    def process_spectra_dataframe_hook(self, df):
        """
        Process the frequency domain dataframe.

        Generally this includes adjusting for geometric spreading,

        masking data under noise levels
        """
        return df

    @_track_method
    def ko_smooth(self, frequencies: Optional[np.ndarray] = None) -> "SpectrumGroup":
        """
        Return new SourceGroup which has konno-ohmachi smoothing applied to it.

        Parameters
        ----------
        frequencies
            Frequencies to use to re-sample the array.

        Returns
        -------

        """
        # TODO add other types of smoothing
        # get inputs for smoothing
        vals = self.data.values
        freqs = self.data.columns.values
        smoothed = ko_smooth(
            vals, frequencies=freqs, center_frequencies=frequencies, normalize=True
        )
        freqs_out = frequencies if frequencies is not None else freqs
        df = pd.DataFrame(smoothed, index=self.data.index, columns=freqs_out)
        return self.new_from_dict(data=df)

    @_track_method
    def subtract_phase(
        self, phase_hint: str = "Noise", drop: bool = True, negative_nan=True
    ) -> "SpectrumGroup":
        """
        Return new SourceGroup with one phase subtracted from the others.

        Parameters
        ----------
        phase_hint
            The phase to subtract. By default use noise.
        drop
            If True drop the subtracted phase, otherwise all its rows will be
            0s.
        negative_nan
            If True set all values below 0 to NaN.
        """
        # get inputs for smoothing
        assert phase_hint in self.data.index.get_level_values("phase_hint")
        subtractor = self.data.loc[phase_hint]
        out = []
        names = []
        for phase_name, df in self.data.groupby(level="phase_hint"):
            # TODO is there such thing as a left subtract?
            if phase_name == phase_hint and drop:
                continue
            ddf = df.loc[phase_name]
            out.append(ddf - subtractor.loc[ddf.index])
            names.append(phase_name)
        df = pd.concat(out, keys=names, names=["phase_hint"])
        if negative_nan:
            df[df < 0] = np.NaN
        return self.new_from_dict(data=df)

    @_track_method
    def mask_by_phase(
        self, phase_hint: str = "Noise", multiplier=1, drop=True
    ) -> "SpectrumGroup":
        """
        Return new SourceGroup masked against another.

        This essentially compares a phase with all other pertinent data and
        masks all values where the first phase is less than the second with NaN.

        By default this will set all values in the signal phases to NaN if they
        are below the noise.

        Parameters
        ----------
        phase_hint
            The phase to subtract. By default use noise.
        multiplier
            A value to multiply the mask by. For example, this can be used
            to mask all values less than 2x the noise.
        drop
            If True drop the subtracted phase, otherwise all its rows will be
            0s.
        """
        # get inputs for smoothing
        assert phase_hint in self.data.index.get_level_values("phase_hint")
        masker = self.data.loc[phase_hint] * multiplier
        out = []
        names = []
        for phase_name, df in self.data.groupby(level="phase_hint"):
            if phase_name == phase_hint and drop:
                continue
            ddf = df.loc[phase_name]
            out.append(ddf.mask(ddf <= masker.loc[ddf.index]))
            names.append(phase_name)
        data = pd.concat(out, keys=names, names=["phase_hint"])
        return self.new_from_dict(data=data)

    @_track_method
    def normalize(self, by="station"):
        """
        Normalize phases of df as if they contained the same number of samples.

        This normalization is necessary because the spectra are taken from
        time windows of different lengths. Without normalization
        this results in longer phases being potentially larger than if they all
        contained the same number of samples before zero-padding.

        Parameters
        ----------
        by
            Some useful description.
        """
        # TODO check if this is correct (may be slightly off)
        df = self.data
        assert df.index.names == _INDEX_NAMES
        # get proper normalization factor for each row
        meta = self.stats.loc[df.index]
        group_col = meta[by]
        tw1, tw2 = meta["starttime"], meta["endtime"]
        samps = ((tw2 - tw1) * self.stats.sampling_rate).astype(int)
        min_samps = group_col.map(samps.groupby(group_col).min())
        norm_factor = (min_samps / samps) ** (1 / 2.0)
        # apply normalization factor
        normed = self.data.mul(norm_factor, axis=0)
        return self.new_from_dict(data=normed)

    @_track_method
    def correct_attenuation(self, quality_facotor=None, drop=True):
        """
        Correct the spectra for intrinsic attenuation.

        Parameters
        ----------
        drop
            If True drop all NaN rows (eg Noise phases)
        """
        df, meta = self.data, self.stats
        required_columns = {"velocity", "quality_factor", "distance"}
        assert set(meta.columns).issuperset(required_columns)
        if quality_facotor is None:
            quality_facotor = meta["quality_factor"]
        # get vectorized q factors
        num = np.pi * meta["distance"]
        denom = quality_facotor * meta["velocity"]
        f = df.columns.values
        factors = np.exp(-np.outer(num / denom, f))
        # apply factors to data
        out = df / factors
        # drop NaN if needed
        if drop:
            out = out[~out.isnull().all(axis=1)]
        return self.new_from_dict(data=out)

    @_track_method
    def correct_radiation_pattern(self, radiation_pattern=None, drop=True):
        """
        Correct for radiation pattern.

        Parameters
        ----------
        radiation_pattern
            A radiation pattern coefficient or broadcastable to data. If None
            uses the default.
        drop
            If True drop any rows without a radiation_pattern.

        Notes
        -----
        By default the radiation pattern coefficient for noise is 1, so the
        noise phases will propagate unaffected.
        """
        if radiation_pattern is None:
            radiation_pattern = self.stats["radiation_coefficient"]
        df = self.data.divide(radiation_pattern, axis=0)
        if drop:
            df = df[~df.isnull().all(axis=1)]
        return self.new_from_dict(data=df)

    @_track_method
    def correct_free_surface(self, free_surface_coefficient=None):
        """
        Correct for stations being on a free surface.

        If no factor is provided uses the one in channel_info.

        Parameters
        ----------
        free_surface_coefficient
        """
        if free_surface_coefficient is None:
            free_surface_coefficient = self.stats["free_surface_coefficient"]
        df = self.data.multiply(free_surface_coefficient, axis=0)
        return self.new_from_dict(data=df)

    @_track_method
    def correct_spreading(self, spreading_coefficient=None):
        """
        Correct for geometric spreading.
        """

        if spreading_coefficient is None:
            spreading_coefficient = self.stats["spreading_coefficient"]

        df = self.data.multiply(spreading_coefficient, axis=0)
        return self.new_from_dict(data= df)

    @_track_method
    def to_motion_type(self, motion_type: str):
        """
        Convert from one ground motion type to another.

        Simply uses spectral division/multiplication and zeros the zero
        frequency. Returns a copy.

        Parameters
        ----------
        motion_type
            Either "displacement", "velocity", or "acceleration".
        """
        # make sure motion type is supported
        if motion_type.lower() not in MOTION_TYPES:
            msg = f"{motion_type} is not in {MOTION_TYPES}"
            raise ValueError(msg)

        current_motion_type = self.motion_type
        if current_motion_type == motion_type:
            return self.copy()

        factor = motion_maps[(current_motion_type, motion_type)]
        df = self.data * factor(self.data.columns)
        # zero freq. 0 in order to avoid inf
        df[0] = 0
        # make new stats object and return
        stats_group = self.stats_group.add_columns(motion_type=motion_type)
        return self.new_from_dict(data=df, stats_group=stats_group)

    @_track_method(idempotent=True)
    def log_resample_spectra(self, length: int):

        # get freqs from dataframe
        freqs = self.data.columns.values

        if length > len(freqs):
            msg = f" length of {length} is higher than number of frequencies"
            raise ValueError(msg)

        # get values from dataframe
        vals = self.data.values
        # resample the frequencies logarithmically
        f_re = np.logspace(np.log10(freqs[1]), np.log10(freqs[-1]), length)
        # use linear interpolation to resample values to the log-sampled frequecies
        interpd = interp.interp1d(freqs, vals)
        # scipy's interpolation function returns a function to interpolate
        vals_re = interpd(f_re)  # return values at the given frequency values
        # re-insert values back into a dataframe
        df = pd.DataFrame(vals_re, index=self.data.index, columns=f_re)

        return self.new_from_dict(data=df)

    # --- functions model fitting

    @_track_method
    def fit_source_model(self, model="brune", fit_noise=False, **kwargs):
        """
        Fit the spectra to a selected source model.

        For more fine-grained control over the inversion see the sourcemodels
        module.

        Parameters
        ----------
        model
            The model, or sequence of models, to fit to the data
        fit_noise
            If True, also fit to the noise, else drop before fitting.

        Returns
        -------

        """
        fit = fit_model(self, model=model, fit_noise=fit_noise, **kwargs)
        return self.new_from_dict(fit_df=fit)

    @_track_method
    def calc_simple_source(self):
        """
        Calculate source parameters from the spectra directly.

        Rather than fitting a model this simply assumes fc is the frequency
        at which the max of the velocity spectra occur and omega0 is the mean
        of all values less than fc.

        The square of the velocity spectra is also added to the source_df.

        These are added to the source df with the group: "simple"
        """
        sg = self.abs()
        # warn/ if any of the pre-processing steps have not occurred
        self._warn_on_missing_process()  # TODO maybe this should raise?
        # get displacement and velocity spectra
        disp = sg.to_motion_type("displacement")
        vel = sg.to_motion_type("velocity")
        # estimate fc as max of velocity (works better when smoothed of course)
        fc = vel.data.idxmax(axis=1)
        lt_fc = np.greater.outer(fc, disp.data.columns.values)
        # create mask of NaN for any values greater than fc
        mask = lt_fc.astype(float)
        mask[~lt_fc] = np.NaN
        # apply mask and get mean excluding NaNs, get moment
        omega0 = pd.Series(np.nanmedian(disp.data * mask, axis=1), index=fc.index)
        density = sg.stats["density"]
        velocity = sg.stats["velocity"]  # TODO change this to "source_velocity"
        moment = omega0 * 4 * np.pi * velocity ** 3 * density
        # calculate velocity_square_sum then normalize by frequency
        sample_spacing = np.median(vel.data.columns[1:] - vel.data.columns[:-1])
        # TODO this should work since we already divide by sqrt(N), check into it
        vel_sum = (vel.data ** 2).sum(axis=1) / sample_spacing
        energy = 4 * np.pi * vel_sum * density * velocity
        # create output source df
        df = pd.concat([omega0, fc, moment, energy], axis=1)
        cols = ["omega0", "fc", "moment", "energy"]
        names = ("method", "parameter")
        df.columns = pd.MultiIndex.from_product([["maxmean"], cols], names=names)
        # add moment magnitude
        mw = (2 / 3) * np.log10(df[("maxmean", "moment")]) - 6.0
        df[("maxmean", "mw")] = mw

        return self.new_from_dict(source_df=self._add_to_source_df(df))

    def _warn_on_missing_process(
        self, spreading=True, attenuation=True, radiation_pattern=True
    ):
        """ Issue warnings if various spectral corrections have not been issued. """
        base_msg = (
            f"calculating source parameters for {self} but "
            f"%s has not been corrected"
        )
        if radiation_pattern and not self.radiation_pattern_corrected:
            warnings.warn(base_msg % "radiation_pattern")
        if spreading and not self.spreading_corrected:
            warnings.warn(base_msg % "geometric spreading")
        if attenuation and not self.attenuation_corrected:
            warnings.warn(base_msg % "attenuation")
        return

    def _add_to_source_df(self, df):
        """
        Adds a dataframe of source parameters to the results dataframe.
        Returns a new dataframe. Will overwrite any existing columns with the
        same names.
        """
        current = self.source_df.copy()
        current[df.columns] = df
        # make sure the proper multi-index is set for columns
        if not isinstance(current.columns, pd.MultiIndex):
            current.columns = pd.MultiIndex.from_tuples(df.columns.values)
        return current

    # -------- Plotting functions

    def plot(
        self,
        event_id: Union[str, int],
        limit=None,
        stations: Optional[Union[str, int]] = None,
        show=True,
    ):
        """
        Plot a particular event id and scaled noise spectra.

        Parameters
        ----------
        event_id
            The event id (str) or event index (as stored in the SourceGroup).
            For example, 0 would return the first event stored in the df.
        stations
            The stations to plot
        limit
            If not None, only plot this many stations.
        show
            If False just return axis for future plotting.
        """
        from mopy.plotting import PlotEventSpectra

        event_spectra_plot = PlotEventSpectra(self, event_id, limit)
        return event_spectra_plot.show(show)

    def plot_centroid_shift(self, show=True, **kwargs):
        """
        Plot the centroid shift by distance differences for each event.
        """
        from mopy.plotting import PlotCentroidShift

        centroid_plot = PlotCentroidShift(self, **kwargs)
        return centroid_plot.show(show)

    def plot_time_domain(
        self,
        event_id: Union[str, int],
        limit=None,
        stations: Optional[Union[str, int]] = None,
        show=True,
    ):
        from mopy.plotting import PlotTimeDomain

        tdp = PlotTimeDomain(self, event_id, limit)
        return tdp.show(show)

    def plot_source_fit(
        self,
        event_id: Union[str, int],
        limit=None,
        stations: Optional[Union[str, int]] = None,
        show=True,
    ):
        from mopy.plotting import PlotSourceFit

        tdp = PlotSourceFit(self, event_id, limit)
        return tdp.show(show)

    # --- utils

    @property
    def spreading_corrected(self):
        """
        Return True if geometric spreading has been corrected.
        """
        return self.in_processing(self.correct_spreading.__name__)

    @property
    def attenuation_corrected(self):
        """
        Return True if attenuation has been corrected.
        """
        return self.in_processing(self.correct_attenuation.__name__)

    @property
    def radiation_pattern_corrected(self):
        """
        Return True if the radiation pattern has been corrected.
        """
        return self.in_processing(self.correct_radiation_pattern.__name__)


motion_maps = {
    ("velocity", "displacement"): lambda freqs: 1 / (2 * np.pi * freqs),
    ("velocity", "acceleration"): lambda freqs: 2 * np.pi * freqs,
    ("acceleration", "velocity"): lambda freqs: 1 / (2 * np.pi * freqs),
    ("acceleration", "displacement"): lambda freqs: 1 / ((2 * np.pi * freqs) ** 2),
    ("displacement", "velocity"): lambda freqs: 2 * np.pi * freqs,
    ("displacement", "acceleration"): lambda freqs: (2 * np.pi * freqs) ** 2,
}
