"""
Class for freq. domain data in mopy.
"""
import warnings
from typing import Optional, Union
from typing_extensions import Literal


import numpy as np
import pandas as pd
import scipy.interpolate as interp

import mopy
import mopy.utils.fft
from mopy.constants import (
    _INDEX_NAMES,
    MOTION_TYPES,
    MOPY_SPECIFIC_DTYPES,
    BroadcastableFloatType,
)
from mopy.core.base import DataGroupBase
from mopy.utils.smooth import konno_ohmachi_smoothing as ko_smooth
from mopy.sourcemodels import fit_model
from mopy.utils.misc import _track_method, _get_alert_function


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

    def __init__(
        self,
        data: pd.DataFrame,
        stats_group: "mopy.StatsGroup",
        spectra_type: str = "dft",
    ):
        super().__init__(stats_group)
        if not set(self.stats.columns).issuperset(MOPY_SPECIFIC_DTYPES):
            msg = (
                "MoPy specific parameters have not been applied to the "
                "StatsGroup. Applying default parameters"
            )
            warnings.warn(msg)
            self.stats_group.apply_defaults(inplace=True)
        # set stats
        self.data = data.copy()
        # init empty source dataframe
        self.source_df = pd.DataFrame(index=self.stats.index)
        self.spectra_type = spectra_type

    # --- SpectrumGroup hooks

    # TODO: Do we need these hooks? It doesn't look like they are used anywhere
    def post_source_function_hook(self):
        """A hook that gets called after each source function."""

    def process_spectra_dataframe_hook(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the frequency domain dataframe.

        Generally this includes adjusting for geometric spreading,

        masking data under noise levels
        """
        return df

    @_track_method
    def ko_smooth(self, frequencies: Optional[np.ndarray] = None) -> "SpectrumGroup":
        """
        Apply konno-ohmachi smoothing and return new SpectrumGroup.

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

    def smooth(self, smoothing, **kwargs) -> "SpectrumGroup":
        """
        Apply the specified smoothing to the spectrum group.

        Parameters
        ----------
        smoothing
            Smoothing to apply

        Other Parameters
        ----------------
        See parameters for specific smoothing methods for valid kwargs.
        """
        if smoothing:
            try:
                smoothing = getattr(self, smoothing)
            except AttributeError:
                raise ValueError(f"Invalid smoothing: {smoothing}")
            return smoothing(**kwargs)  # This is cramping my brain a little bit
        else:
            return self.copy()

    @_track_method
    def subtract_phase(
        self, phase_hint: str = "Noise", negative_nan=True
    ) -> "SpectrumGroup":
        """
        Return new SourceGroup with one phase subtracted from the others.

        Parameters
        ----------
        phase_hint
            The phase to subtract. By default use noise.
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

    # @_track_method
    # def normalize(self, by: str = "station") -> "SpectrumGroup":
    #     """
    #     Normalize phases of df as if they contained the same number of samples.
    #
    #     This normalization is necessary because the spectra are taken from
    #     time windows of different lengths. Without normalization
    #     this results in longer phases being potentially larger than if they all
    #     contained the same number of samples before zero-padding.
    #
    #     Parameters
    #     ----------
    #     by
    #         Grouping to apply to the data
    #     """
    #     # TODO check if this is correct (may be slightly off)
    #     df = self.data
    #     assert df.index.names == _INDEX_NAMES
    #     # get proper normalization factor for each row
    #     meta = self.stats.loc[df.index]
    #     group_col = meta[by]
    #     tw1, tw2 = meta["starttime"], meta["endtime"]
    #     samps = ((tw2 - tw1) * self.stats.sampling_rate).astype(int)
    #     min_samps = group_col.map(samps.groupby(group_col).min())
    #     norm_factor = (min_samps / samps) ** (1 / 2.0)
    #     # apply normalization factor
    #     normed = self.data.mul(norm_factor, axis=0)
    #     return self.new_from_dict(data=normed)

    @_track_method
    def to_spectra_type(self, spectra_type: str) -> "SpectrumGroup":
        """
        Convert the data to the specified spectra type (ex., discrete fourier
        transform to power spectral density)

        Parameters
        ----------
        spectra_type
            Spectra type to convert to
        """
        current_type = self.spectra_type
        if current_type == spectra_type:
            return self.new_from_dict(data=self.data.copy())
        try:
            conversion = getattr(mopy.utils.fft, f"{current_type}_to_{spectra_type}")
        except AttributeError:
            raise ValueError(f"Invalid spectra type: {spectra_type}")
        assert self.data.index.names == _INDEX_NAMES
        df = conversion(
            self.data, sampling_rate=self.sampling_rate, npoints=self.stats["npts"]
        )
        return self.new_from_dict(data=df, spectra_type=spectra_type)

    @_track_method
    def correct_attenuation(
        self, quality_factor: Optional[BroadcastableFloatType] = None
    ) -> "SpectrumGroup":
        """
        Correct the spectra for intrinsic attenuation.

        Parameters
        ----------
        quality_factor
            Quality factor to use for the attenuation correction

        Notes
        -----
        By default the quality factor for noise is 1e9, to prevent meaningful
        attenuation
        """
        df, meta = self.data, self.stats.loc[self.data.index]
        required_columns = {"source_velocity", "quality_factor", "ray_path_length_m"}
        assert set(meta.columns).issuperset(required_columns)
        if quality_factor is None:
            quality_factor = meta["quality_factor"]
        # get vectorized q factors
        num = np.pi * meta["ray_path_length_m"]
        denom = quality_factor * meta["source_velocity"]
        f = df.columns.values
        factors = np.exp(-np.outer(num / denom, f))
        # apply factors to data
        out = df / factors
        # drop NaN if needed
        return self.new_from_dict(data=out)

    @_track_method
    def correct_radiation_pattern(
        self, radiation_pattern: Optional[BroadcastableFloatType] = None
    ) -> "SpectrumGroup":
        """
        Correct for radiation pattern.

        Parameters
        ----------
        radiation_pattern
            A radiation pattern coefficient or broadcastable to data. If None
            uses the default.

        Notes
        -----
        By default the radiation pattern coefficient for noise is 1, so the
        noise phases will propagate unaffected.
        """
        if radiation_pattern is None:
            radiation_pattern = self.stats["radiation_coefficient"]
        df = self.data.divide(radiation_pattern, axis=0)
        return self.new_from_dict(data=df)

    @_track_method
    def correct_free_surface(
        self, free_surface_coefficient: Optional[BroadcastableFloatType] = None
    ) -> "SpectrumGroup":
        """
        Correct for stations being on a free surface.

        If no factor is provided uses the one in channel_info.

        Parameters
        ----------
        free_surface_coefficient
            Free surface correction to apply. If None, uses the default.
        """
        if free_surface_coefficient is None:
            free_surface_coefficient = self.stats["free_surface_coefficient"]
        df = self.data.multiply(free_surface_coefficient, axis=0)
        return self.new_from_dict(data=df)

    @_track_method
    def correct_spreading(
        self, spreading_coefficient: Optional[BroadcastableFloatType] = None
    ) -> "SpectrumGroup":
        """
        Correct for geometric spreading.

        Parameters
        ----------
        spreading_coefficient
            Geometric spreading correction to apply. If None, uses the default.

        Notes
        -----
        By default the spreading coefficient for noise is 1, so the
        noise phases will propagate unaffected.
        """

        if spreading_coefficient is None:
            spreading_coefficient = self.stats["spreading_coefficient"]

        df = self.data.divide(spreading_coefficient, axis=0)
        return self.new_from_dict(data=df)

    def apply_default_corrections(self) -> "SpectrumGroup":
        """
        Convenience function to apply the default corrections.

        Performs the following:
            1. get abs of spectra
            2. correct for radiation pattern
            3. correct for geometric spreading
            4. correct for attenuation
            5. correct for free surface
        """
        sg = (
            self.abs()
            .correct_radiation_pattern()
            .correct_spreading()
            .correct_attenuation()
            .correct_free_surface()
        )
        return sg

    @_track_method
    def to_motion_type(self, motion_type: str) -> "SpectrumGroup":
        """
        Convert from one ground motion type to another.

        The ground motion conversion is done in the time domain. Returns a copy.

        Parameters
        ----------
        motion_type
            Motion type to convert to. Allowable values include:
            "displacement", "velocity", or "acceleration"
        """
        # Make sure motion type is supported
        if motion_type.lower() not in MOTION_TYPES:
            msg = f"{motion_type} is not in {MOTION_TYPES}"
            raise ValueError(msg)

        current_motion_type = self.motion_type
        if current_motion_type == motion_type:
            return self.copy()
        try:
            conversion = motion_maps[(current_motion_type, motion_type)]
        except KeyError:
            raise ValueError(f"Invalid motion type: {motion_type}")

        # Change over to the time domain
        td = np.fft.irfft(self.data, axis=-1)
        # Do the conversion
        conv = conversion(td, self.sampling_rate)
        # Change back to the frequency domain
        fd = np.fft.rfft(conv, axis=-1)

        # Rebuild the dataframe
        df = pd.DataFrame(data=fd, columns=self.data.columns, index=self.data.index)
        # make a new StatsGroup with the motion type and return
        stats_group = self.stats_group.add_columns(motion_type=motion_type)
        return self.new_from_dict(data=df, stats_group=stats_group)

    @_track_method(idempotent=True)
    def log_resample_spectra(self, length: int) -> "SpectrumGroup":
        """
        Apply a logarithmic resampling of the spectra

        Parameters
        ----------
        length
            Number of points for the resampled frequencies
        """

        # get freqs from dataframe
        freqs = self.data.columns.values

        if length > len(freqs):
            msg = f" length of {length} is higher than number of frequencies"
            raise ValueError(msg)

        # get values from dataframe
        vals = self.data.values
        # resample the frequencies logarithmically
        f_re = np.logspace(np.log10(freqs[1]), np.log10(freqs[-1]), length)
        # use linear interpolation to resample values to the log-sampled frequencies
        interpd = interp.interp1d(freqs, vals)
        # scipy's interpolation function returns a function to interpolate
        vals_re = interpd(f_re)  # return values at the given frequency values
        # re-insert values back into a dataframe
        df = pd.DataFrame(vals_re, index=self.data.index, columns=f_re)

        return self.new_from_dict(data=df)

    # --- functions model fitting

    @_track_method
    def fit_source_model(
        self, model: str = "brune", fit_noise: bool = False, **kwargs
    ) -> "SpectrumGroup":
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
        """
        fit = fit_model(self, model=model, fit_noise=fit_noise, **kwargs)
        return self.new_from_dict(fit_df=fit)

    def calc_corner_frequency(self) -> pd.Series:
        """
        Calculate the corner frequency of the displacement spectra

        Notes
        -----
        Calculation is actually performed by getting the maximum of the velocity spectra
        """
        sg = self.abs()
        # Convert to velocity spectra
        vel = sg.to_motion_type("velocity").to_spectra_type("cft")

        return vel.data.idxmax(axis=1)  # Corner frequency

    def calc_omega0(self, fc: pd.Series) -> pd.Series:
        """
        Calculate Omega0 for the displacement spectra

        Parameters
        ----------
        fc
            Corner frequencies for each trace
        """
        sg = self.abs()
        disp = sg.to_motion_type("displacement").to_spectra_type("cft")
        # Exclude values greater than the corner frequency
        gt_fc = np.greater.outer(fc.values, disp.data.columns.values)
        mask = gt_fc.astype(float)
        mask[~gt_fc] = np.NaN
        # Calculate omega0, ignoring the NaN values
        return pd.Series(np.nanmedian(disp.data * mask, axis=1), index=fc.index)

    # @_track_method
    def _calc_spectral_params(self) -> pd.DataFrame:
        """
        Calculate spectral parameters from the spectra directly.

        Rather than fitting a model this simply assumes fc is the frequency
        at which the max of the velocity spectra occur and omega0 is the mean
        of all values less than fc.

        The square of the velocity spectra is also added to the source_df.

        """
        sg = self.abs()
        # get displacement spectra and velocity power density spectra
        # estimate fc as max of velocity (works better when smoothed of course)
        fc = sg.calc_corner_frequency()
        fc_per_station = fc.groupby(["phase_hint", "event_id", "seed_id_less"]).mean()
        fc_per_station.name = "fc"
        # estimate omega0
        omega0 = sg.calc_omega0(fc)
        omega0_per_station = omega0.groupby(
            ["phase_hint", "event_id", "seed_id_less"]
        ).apply(np.linalg.norm)
        omega0_per_station.name = "omega0"
        return pd.concat([fc_per_station, omega0_per_station], axis=1)

    def calc_moment(self, omega0: pd.Series) -> pd.Series:
        """
        Calculate the seismic moment.

        Parameters
        ----------
        omega0
            List of omega0 calculated for each trace

        Returns
        -------
        Seismic moment in N-m. All components of a station will be combined
        for each event/pick.
        """
        # Get necessary values from the stats dataframe
        density = (
            self.stats["density"]
            .groupby(["phase_hint", "event_id", "seed_id_less"])
            .mean()
        )
        source_velocity = (
            self.stats["source_velocity"]
            .groupby(["phase_hint", "event_id", "seed_id_less"])
            .mean()
        )
        # Get moment
        moment = omega0 * 4 * np.pi * source_velocity**3 * density
        moment.name = "moment"
        return moment

    def calc_moment_mag(self, moment: pd.Series) -> pd.Series:
        """
        Calculate moment magnitude

        Parameters
        ----------
        moment
            Seismic moment (N-m) calculated for each station/phase/event
        """
        mw = 2.0 / 3.0 * (np.log10(moment) - 9.1)
        mw.name = "mw"
        return mw

    def calc_potency(self, omega0: pd.Series) -> pd.Series:
        """
        Calculate the seismic potency

        Parameters
        ----------
        omega0
            List of omega0 calculated for each trace

        Returns
        -------
        Seismic potency in m^2/s^2. All components of a station will be combined
        for each event/pick.
        """
        source_velocity = (
            self.stats["source_velocity"]
            .groupby(["phase_hint", "event_id", "seed_id_less"])
            .mean()
        )
        potency = (
            omega0 * 4 * np.pi * source_velocity
        )  # The other parameters should have already been corrected
        potency.name = "potency"
        return potency

    def calc_energy(self) -> pd.Series:
        """
        Calculate the radiated seismic energy

        Returns
        -------
        Radiated seismic energy in Joules. All components of a station will be
        combined for each event/pick.
        """
        sg = self.abs()
        # Get the velocity psd and integrate
        vel_psd = sg.to_motion_type("velocity").to_spectra_type("psd")
        vel_psd_per_station = vel_psd.data.groupby(
            ["phase_hint", "event_id", "seed_id_less"]
        ).sum()
        vel_psd_int = vel_psd_per_station.sum(axis=1)
        # Get necessary values from the stats dataframe
        density = (
            self.stats["density"]
            .groupby(["phase_hint", "event_id", "seed_id_less"])
            .mean()
        )
        source_velocity = (
            self.stats["source_velocity"]
            .groupby(["phase_hint", "event_id", "seed_id_less"])
            .mean()
        )
        # Calculate the energy
        energy = 4 * np.pi * vel_psd_int * density * source_velocity
        energy.name = "energy"
        return energy

    # @_track_method
    def calc_source_params(self, enforce_preprocessing=True) -> pd.DataFrame:
        """
        Calculate the source parameters.

        Currently, source params include the following:
            (omega0, fc, moment, potency, energy, and mw).

        Corrections should have been previously applied to the SpectraGroup
        or a ValueError will be raised unless disabled with
        enforce_preprocessing.

        Parameters
        ----------
        enforce_preprocessing
            If True, raise a ValueError if corrections were not already applied.
        """
        # First calculate the spectral parameters
        mode = "raise" if enforce_preprocessing else "ignore"
        self.check_corrected(mode=mode)
        source_df = self._calc_spectral_params()
        # Calculate the various source parameters
        moment = self.calc_moment(source_df["omega0"])
        mw = self.calc_moment_mag(moment)
        potency = self.calc_potency(source_df["omega0"])
        sg = self.abs()
        energy = sg.calc_energy()
        source_df = pd.concat([source_df, moment, potency, energy, mw], axis=1)
        # drop rows with all nan and noise phases
        out = source_df[
            source_df.index.get_level_values("phase_hint") != "Noise"
        ].dropna(how="all", axis=0)
        return out

    def check_corrected(
        self,
        mode: Literal["warn", "ignore", "raise"] = "warn",
        spreading: bool = True,
        attenuation: bool = True,
        radiation_pattern: bool = True,
        free_surface: bool = True,
    ) -> None:
        """
        Issue warnings if various spectral corrections have not been issued.
        """
        base_msg = (
            f"calculating source parameters for {self} but "
            f"%s has not been corrected"
        )
        func = _get_alert_function(mode)
        if radiation_pattern and not self.radiation_pattern_corrected:
            func(base_msg % "radiation_pattern")
        if spreading and not self.spreading_corrected:
            func(base_msg % "geometric spreading")
        if attenuation and not self.attenuation_corrected:
            func(base_msg % "attenuation")
        if free_surface and not self.free_surface_corrected:
            func(base_msg % "free surface")
        return

    def _add_to_source_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a dataframe of source parameters to the results dataframe.
        Returns a new dataframe. Will overwrite any existing columns with the
        same names.
        """
        current = self.source_df.copy()
        current[df.columns] = df
        # Todo: Once again, not going to worry about this multiindex for right now
        # # make sure the proper multi-index is set for columns
        # if not isinstance(current.columns, pd.MultiIndex):
        #     current.columns = pd.MultiIndex.from_tuples(df.columns.values)
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
        from mopy.utils.plotting import PlotEventSpectra

        event_spectra_plot = PlotEventSpectra(self, event_id, limit)
        return event_spectra_plot.show(show)

    def plot_centroid_shift(self, show: bool = True, **kwargs):
        """
        Plot the centroid shift by distance differences for each event.
        """
        from mopy.utils.plotting import PlotCentroidShift

        centroid_plot = PlotCentroidShift(self, **kwargs)
        return centroid_plot.show(show)

    def plot_time_domain(
        self,
        event_id: Union[str, int],
        limit=None,
        stations: Optional[Union[str, int]] = None,
        show=True,
    ):
        """Plot the data in the time domain"""
        from mopy.utils.plotting import PlotTimeDomain

        tdp = PlotTimeDomain(self, event_id, limit)
        return tdp.show(show)

    def plot_source_fit(
        self,
        event_id: Union[str, int],
        limit=None,
        stations: Optional[Union[str, int]] = None,
        show=True,
    ):
        """Plot the fit of the spectral data"""
        from mopy.utils.plotting import PlotSourceFit

        tdp = PlotSourceFit(self, event_id, limit)
        return tdp.show(show)

    # --- utils

    @property
    def spreading_corrected(self) -> bool:
        """
        Return True if geometric spreading has been corrected.
        """
        return self.in_processing(self.correct_spreading.__name__)

    @property
    def attenuation_corrected(self) -> bool:
        """
        Return True if attenuation has been corrected.
        """
        return self.in_processing(self.correct_attenuation.__name__)

    @property
    def radiation_pattern_corrected(self) -> bool:
        """
        Return True if the radiation pattern has been corrected.
        """
        return self.in_processing(self.correct_radiation_pattern.__name__)

    @property
    def free_surface_corrected(self) -> bool:
        """
        Return True if free surface has been corrected.
        """
        return self.in_processing(self.correct_free_surface.__name__)


def differentiate_time(data: np.array, sample_rate: float) -> np.array:
    """
    Differentiate an array of time series

    Parameters
    ----------
    data
        DataFrame of time series data to differentiate
    sample_rate
        Sample rate of the data
    """
    return np.gradient(data, 1 / sample_rate, axis=-1)


def integrate_time(data: np.array, sample_rate: float) -> np.array:
    """
    Integrate an array of time series

    Parameters
    ----------
    data
        DataFrame of time series data to differentiate
    sample_rate
        Sample rate of the data
    """
    return np.cumsum(data, axis=-1) / sample_rate


motion_maps = {
    ("displacement", "velocity"): differentiate_time,  # Differentiate once
    ("displacement", "acceleration"): lambda data, sr: differentiate_time(
        differentiate_time(data, sr), sr
    ),  # Differentiate twice
    ("velocity", "acceleration"): differentiate_time,  # Differentiate once
    ("velocity", "displacement"): integrate_time,  # Integrate once
    ("acceleration", "velocity"): integrate_time,  # Integrate once
    ("acceleration", "displacement"): lambda data, sr: integrate_time(
        integrate_time(data, sr), sr
    ),  # Integrate twice
}
