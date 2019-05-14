"""
Some classes for grouping together plotting functionality.
"""
import inspect
from functools import partial
from itertools import cycle
from itertools import product
from types import MappingProxyType as MapProxy

import matplotlib.pyplot as plt
import numpy as np
from obsplus.utils import iterate

DEFAULT_COLORS = ("b", "r", "g", "y", "k")


class VerticalWithSubPlots:
    """
    Base class for any vertical plots that can have any number of subplots.
    """

    inch_per_subplot = 3
    width = 5
    degree_char = "\u00B0"

    def __init__(self):
        plt.clf()
        plt.close()

    def show(self, show):
        """ either save the fig, call plt.show, or do nothing. """
        if isinstance(show, str):
            plt.savefig(show)
        elif show:
            plt.show()

    def savefig(self, path):
        plt.savefig(path)

    def _init_figure(self, num_stations, num_channels):
        """ initialize subplots/figure. """

        size = (self.width * num_channels, self.inch_per_subplot * num_stations)

        fig, subplots = plt.subplots(
            num_stations, num_channels, sharex=True, sharey=True, figsize=size
        )
        # ensure subplots is iterable
        if not isinstance(subplots, np.ndarray):
            subplots = np.array([subplots])
        return fig, np.atleast_2d(subplots)

    def _get_filtered_df(self, df, event_id):
        """ return the filtered dataframe. """
        # filter df to only include data of interest
        slice_tuple = (slice(None), slice(event_id), slice(None))
        # make sure only one event_is_selected
        correct_eid = df.index.get_level_values("event_id") == event_id
        return df[correct_eid].loc[slice_tuple, :]

    def _get_event_id(self, df, event_id):
        event_ids = df.index.get_level_values("event_id").unique()
        if isinstance(event_id, str):
            assert event_id in event_ids
        elif isinstance(event_id, int):
            assert event_id < len(event_ids)
            event_id = event_ids[event_id]
        return event_id

    def _get_subplot_title(self, dist_df, event_id, sta_df, sta):
        """ Get the title for a subplot. """
        # get distance, azimuth
        ser = dist_df.loc[(event_id, sta_df.index[0][-1])]
        dist_km = ser.distance / 1000.0
        azimuth = ser.azimuth
        return f"{sta} [{dist_km:0.2f}km, {azimuth:0.2f}{self.degree_char}]"

    def _get_axis_dict(self, meta, event_id, limit, stations):
        """
        get a dictionary of axis.

        {(station: channl): axis}
        """
        # select out desired event
        meta = meta.xs(event_id, level="event_id")
        # sort based on distance
        meta = meta.sort_values("distance")
        # filter out stations
        if stations:  # if a particular set of stations was selected
            # TODO add matching here (maybe?)
            stations = list(iterate(stations))
        else:
            stations = np.unique(meta["station"])
        if limit:
            stations = stations[:limit]
        meta = meta[meta.station.isin(stations)]
        # get station channel matrix
        pcols = ["station", "channel"]
        piv = meta.drop_duplicates(pcols).pivot(
            values="distance", index="station", columns="channel"
        )
        num_chan = len(piv.columns)
        num_sta = len(piv.index)
        fig, axis = self._init_figure(num_sta, num_chan)
        # create dict
        out = {}
        for chan_num, sta_num in product(range(num_chan), range(num_sta)):
            sta_name, chan_name = piv.index[sta_num], piv.columns[chan_num]
            out[(sta_name, chan_name)] = axis[sta_num][chan_num]

        return fig, out


class PlotCentroidShift(VerticalWithSubPlots):
    """ Plots the shift in centroid. """

    def __init__(self, source_group, dist_col="distance", plot_stations=False):
        super().__init__()
        # get meta data, assert centroid freqs are there and filter out noise
        df = source_group.meta
        phase_hint = df.index.get_level_values("phase_hint")
        df = df.loc[phase_hint != "Noise"]
        phase_hint = df.index.get_level_values("phase_hint")
        # make figure and subplots
        fig, subplots = self._init_figure(len(phase_hint.unique()))
        event_count = len(df.index.get_level_values("event_id").unique())
        colors = [c for _, c in zip(range(event_count), cycle(DEFAULT_COLORS))]
        # iterate phases, make subplot for each. each event gets different color
        for num, (phase, pdf) in enumerate(df.groupby(phase_hint)):
            axis = subplots[num]  # get panel for this phase
            eids = pdf.index.get_level_values("event_id")
            for enum, (event_id, edf) in enumerate(pdf.groupby(eids)):
                color = colors[enum]
                sd = edf.sort_values(dist_col)
                x, y = sd[dist_col].values / 1000.0, sd.centroid_frequency.values
                # plot lines connecting points
                axis.plot(x, y, "-.", label=event_id, color=color)
                # plot either station names or dots
                if plot_stations:
                    for x_, y_, txt in zip(x, y, sd.station.values):
                        axis.text(x_, y_, txt)
                else:
                    axis.plot(x, y, ".", label=event_id, color=color)
                # calculate line of best fit via linear regression and plot
                poly = np.polyfit(x, y, 1)
                xmax = pdf[dist_col].max() / 1000.0  # dist in km
                axis.plot([0, xmax], [poly[-1], xmax * poly[0] + poly[-1]])

            axis.title.set_text(f"Phase: {phase}")
            axis.set_ylabel("centroid_frequency (Hz)")
        plt.xlabel(f"{dist_col} (km)")


class PlotEventSpectra(VerticalWithSubPlots):
    """" Class for plotting event spectra. """

    colors = {"Noise": "b", "P": "r", "S": "g"}
    _source_funcs = MapProxy({})
    _source_kwargs = MapProxy({})

    def __init__(self, source_group, event_id, limit=None, stations=None):
        super().__init__()
        self.source_group = source_group
        self.freqs = source_group.data.columns
        event_id = self._get_event_id(source_group.data, event_id)
        df = self._get_filtered_df(abs(source_group.data), event_id)
        source_df = source_group.source_df
        # slice meta to only get same selection as df
        meta = source_group.meta.loc[df.index]
        # init a dict of subplots {(phase: seed_id): axis}
        fig, ax_dict = self._get_axis_dict(meta, event_id, limit, stations)

        # init partials of source_models if used
        self._get_source_funcs()

        for (sta, chan), ax in ax_dict.items():
            sub_meta = meta[(meta.station == sta) & (meta.channel == chan)]
            data = df.loc[sub_meta.index]
            self._plot_channel(ax, data, meta)
            # plot fitted dataframe if applicable
            if source_df is not None and not source_df.empty:
                self._plot_fitted_source(ax, source_df.loc[sub_meta.index], sub_meta)

        # iterate all axis and turn on legends
        for ax in ax_dict.values():
            ax.legend(loc=3)

    def _get_source_funcs(self):
        """ set _source_funcs and _source_kwargs """
        # set a few variables; bail out if not fitted df
        sg = self.source_group
        data = sg.data
        source_df = sg.source_df
        if source_df is None or source_df.empty:
            return

        from mopy.sourcemodels import source_spectrum, SOURCE_MODEL_PARAMS

        # get frequencies and model function partials
        freqs = data.columns
        #
        used_models = source_df.columns.get_level_values("model").unique()
        # create dict of partial source model functions
        funcs = {}
        wanted_kwargs = {}
        for model in used_models:
            source_params = SOURCE_MODEL_PARAMS[model]
            func = partial(source_spectrum, freqs=freqs, **source_params)
            funcs[model] = func
            wanted_kwargs[model] = set(inspect.signature(func).parameters)
        self._source_funcs = funcs
        self._source_kwargs = wanted_kwargs

    def _plot_fitted_source(self, ax, fit_df, meta):
        """ plot the fitted source spectra """
        used_models = fit_df.columns.get_level_values("model").unique()
        # filter out nulls
        fit_df = fit_df[~fit_df.isnull().any(axis=1)]

        # iterate each row that is not null
        for ind, row in fit_df.iterrows():
            phase = ind[0]
            color = self.colors[phase]
            for model in used_models:
                wanted_kwargs = self._source_kwargs[model]
                func = self._source_funcs[model]
                # get inputs to particular model function
                meta_dict = dict(meta.loc[ind[:2]])
                kwargs = dict(row.loc[model])
                kwargs.update(meta_dict)
                # get desired inputs from signature
                overlap = wanted_kwargs & set(kwargs)
                kwargs = {x: kwargs[x] for x in overlap}
                # calc spectra
                data = func(**kwargs)
                # get label and plot
                kwargs_str = "; ".join([f"{i}: {v:.2E}" for i, v in kwargs.items()])
                label = f"{model}_{phase}: {kwargs_str}"
                ax.plot(
                    self.freqs, data, color=color, ls="-", label=label, linestyle="--"
                )

        # for model in :
        #     model_params = SOURCE_MODEL_PARAMS[model]
        #     func = partial(source_spectrum, **model_params)
        #     for ind, df in fit_df.iterrows()
        #     time = funq(freqs)
        #
        #
        #     breakpoint()

        sg = self.source_group

    def _plot_channel(self, ax, data, meta):
        """ plot the channel data. """
        for ind, row in data.iterrows():
            meta_row = meta.loc[ind]
            phase = ind[0]
            color = self.colors.get(phase, "k")
            ax.loglog(row.index, row.values, label=phase, color=color)
            ax.set_title(ind[-1])

        # fit_df = self.source_group.fit_df
        # # iterate over each channel/phase in station
        # channels = sta_df.index.get_level_values('seed_id').str[-3:]
        # channel_map = {chan: num for num, chan
        #                in enumerate(sorted(set(channels)))}
        # for ind, row in sta_df.iterrows():
        #     # get axis to plot on
        #     channel_code = ind[-1].split(".")[-1]
        #     chan_num = channel_map[channel_code]
        #     ax = axis[chan_num]
        #     phase_name = ind[0]
        #     color = self.colors.get(phase_name, "k")
        #     label = f"{phase_name}"
        #     ax.loglog(row.index, row.values, label=label, color=color)
        #     ax.set_title(ind[-1])
        #     # get info on source and model for plotting if it is there
        #     if fit_df is not None:
        #         fit = fit_df.loc[ind]
        # for ax in axis:
        #     ax.legend()


# class PlotSourceFit(VerticalWithSubPlots):
#     def __init__(self, source_group, event_id, limit=None, plot_centroid=False):
#         super().__init__()
#         self.plot_centroid = plot_centroid
#         event_id = self._get_event_id(source_group.data, event_id)
#         df = self._get_filtered_df(abs(source_group.data), event_id)
#         # slice meta to only get same selection as df
#         meta = source_group.meta.loc[df.index]
#         # get stations and unique stations
#         stations = meta.station.values
#         unique_stations = np.unique(stations)
#         if limit:
#             unique_stations = unique_stations[:limit]
#         ustations = {sta: num for num, sta in enumerate(unique_stations)}
#         # init figure/subplots
#         fig, subplots = self._init_figure(len(ustations))
#         # iterate over each station and plot
#         for sta, sta_df in df.groupby(stations):
#             if sta not in ustations:  # skip if station not used
#                 continue
#             axis = subplots[ustations[sta]]
#             self._plot_station(axis, sta, sta_df)
#             # show legend, set title
#             axis.legend()
#             args = (source_group.dist, event_id, sta_df, sta)
#             title = self._get_subplot_title(*args)
#             axis.title.set_text(title)
#         # set axis labels
#         plt.xlabel("frequency (Hz)")


class PlotTimeDomain(VerticalWithSubPlots):
    """
    Plot time domain of event rep with subplots.
    """

    colors = {"Noise": "b", "P": "r", "S": "g"}

    def __init__(self, source_group, event_id, limit=None):
        super().__init__()
        event_id = self._get_event_id(source_group.data, event_id)
        df = self._get_filtered_df(source_group._td_data, event_id)
        # slice meta to only get same selection as df
        meta = source_group.meta.loc[df.index]
        # get stations and unique stations
        stations = meta.station.values
        unique_stations = np.unique(stations)
        if limit:
            unique_stations = unique_stations[:limit]
        ustations = {sta: num for num, sta in enumerate(unique_stations)}
        # init figure/subplots
        fig, subplots = self._init_figure(len(ustations))
        # iterate over each station and plot
        for sta, sta_df in df.groupby(stations):
            if sta not in ustations:  # skip if station not used
                continue
            axis = subplots[ustations[sta]]
            self._plot_station(axis, sta, sta_df, meta)

            # show legend, set title
            axis.legend()
            args = (source_group.dist, event_id, sta_df, sta)
            title = self._get_subplot_title(*args)
            axis.title.set_text(title)
        # set axis labels
        plt.xlabel("Time (sec)")

    def _plot_station(self, axis, sta, sta_df, meta):
        """ plot a stations spectra """
        # iterate over each channel/phase in station
        for ind, row in sta_df.iterrows():
            meta_row = meta.loc[ind]
            phase_name = ind[0]
            color = self.colors.get(phase_name, "k")
            channel_code = ind[-1].split(".")[-1]
            label = f"{phase_name}_{channel_code}"
            axis.plot(row.index, row.values, label=label, color=color)
            # draw dotd line on centroid
