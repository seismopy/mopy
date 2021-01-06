"""
A pipeline for processing source params from a local node dataset.

A node dataset typically consists of short-period geophones deployed
on the surface in a dense networking surrounding the seismogenic volume.

Note: Many sensible defaults are selected and this pipeline should
perform reasonable well under most circumstances for local node
networks. If you need more control/customization you will need to
use the classes from the core module.
"""

from typing import Union, Optional, Callable, Dict, Tuple

import numpy as np
import obspy
import pandas as pd
from obsplus.interfaces import WaveformClient
from obsplus.utils.docs import compose_docstring
from obspy import Stream
from obspy.core.event import Catalog, Event, Magnitude, CreationInfo
from obspy import Stream, UTCDateTime


import mopy
from mopy.constants import BroadcastableFloatType
from mopy.utils.misc import SourceParameterAggregator


events_waveforms_params = """
events
    A catalog or Event object
waveforms
    Any object from which waveforms can be extracted.
"""


class LocalNodePipeLine:
    """
    A pipeline suitable for calculating reasonable source parameters for a close in nodal deployment
    (stations within 2 km of events)

    Parameters
    ----------
    inventory
        The inventory of stations which will be used to calculate source
        parameters.
    waveforms
        A waveform client; any object which implements the get_waveforms
        interface.
    quality_factor
        The quality factor used for intrinsic attenuation corrections.
    stream_processor
        A boolean indicating if the default stream processor (which removes
        response and applies a linear detrend) should be used. Can also
        be a callable which takes a single stream as the only argument
        and returns a stream. Either way, one should ensure the instrument
        responses have been removed and the traces de-trended.

    Attributes
    ----------
    prefilt_low
        The lower end of the pre-filter used to stabilize the instrument
        response removal. Does nothing stream_processor is False or a custom
        function is used.
    prefilt_high
        The upper end of the pre-filter used to stabilize the instrument
        response removal. Does nothing stream_processor is False or a custom
        function is used.

    """

    # default prefilters to use for stabilizing instrument response
    # removal. prefilt_high is written in terms of the nyquist freq.
    prefilt_low = (0.2, 0.5)
    prefilt_high = (0.4, 0.5)
    # Map column names from calc_source_params to magnitude types
    _mag_type_map = {
        "mw": "Mw",
        "energy": "log(E)",
        "potency": "log(P)",
    }
    __version__ = "0.0.0"

    def __init__(
        self,
        inventory: obspy.Inventory,
        waveforms: Optional[Union[WaveformClient, Stream]] = None,
        quality_factor: Optional[BroadcastableFloatType] = None,
        source_velocity: Optional[BroadcastableFloatType] = None,
        density: Optional[BroadcastableFloatType] = None,
        stream_processor: Union[bool, Callable[[Stream], Stream]] = True,
    ):
        self._inventory = inventory
        self._waveform_client = waveforms
        self._quality_factor = quality_factor
        self._source_velocity = source_velocity
        self._density = density
        # use custom function for removing response if defined
        self._stream_processor = self._get_stream_processor(stream_processor)
        if stream_processor is None:  # else use default
            self._stream_processor = self._remove_response

    def _get_stream_processor(self, stream_processor):
        """get an appropriate stream processing input"""
        out = None
        if callable(stream_processor):
            out = stream_processor
        elif stream_processor:
            out = self._remove_response
        return out

    @property
    def _method_uri(self):
        """Generate an author URI"""
        name = self.__class__.__name__
        version = self.__version__
        mopy_version = mopy.__version__
        return f"MoPy_{mopy_version}_{name}_{version}"

    def _create_info(self):
        """Make creation info for """
        out = CreationInfo(creation_time=UTCDateTime().now(),)
        return out

    @compose_docstring(ew_params=events_waveforms_params)
    def add_magnitudes(
        self,
        events: Union[Event, Catalog],
        waveforms: Optional[Union[WaveformClient, Stream]] = None,
        mw_preferred: bool = False,
    ) -> obspy.Catalog:
        """
        Calculate and add magnitudes to the event object in-place.

        Parameters
        {ew_params}
        mw_preferred
            If True mark the new Mw as the preferred magnitude
        """
        # index events
        elist = [events] if isinstance(events, Event) else events
        event_index = {str(event.resource_id): event for event in elist}
        # get magnitude dict
        magnitudes = self.create_simple_magnitudes(elist, waveforms)
        # attach magnitudes to appropriate events
        for (eid, mag_type), mag in magnitudes.items():
            event = event_index[eid]
            event.magnitudes.append(mag)
            # mark mw as preferred if requested
            if mw_preferred and mag_type == "mW":
                event.preferred_magnitude_id = event.resource_id
        return events

    @compose_docstring(ew_params=events_waveforms_params)
    def create_simple_magnitudes(
        self,
        events: Union[Event, Catalog],
        waveforms: Optional[Union[WaveformClient, Stream]] = None,
    ) -> Dict[Tuple[str, str], Magnitude]:
        """
        Create magnitude objects from calculated source params.

        Returns a dict of {(event_id, magnitude_type): Magnitude}

        Parameters
        ----------
        {ew_params}
        """
        df = self.calc_source_parameters(events, waveforms)
        eids = set(df.index)
        out = {}
        for eid in eids:
            for col_name, mag_type in self._mag_type_map.items():
                log = "log" in mag_type
                value = df.loc[eid, col_name]
                mag = Magnitude(  # Need to flesh this out to make sure it has all of the components that can reasonably be filled in (ex., author and agency are missing from the creation info)
                    magnitude_type=mag_type,
                    method_id=self._method_uri,
                    creation_info=self._create_info(),
                    mag=np.log10(value) if log else value,
                )
                out[(eid, mag_type)] = mag
        return out

    def calc_source_parameters(
        self,
        events: Union[Event, Catalog],
        waveforms: Optional[Union[WaveformClient, Stream]] = None,
    ) -> pd.DataFrame:
        """
        Calculate the source parameters for each event.

        This just uses :meth:`LocalNodePipeline.calc_station_source_parameters`
        then aggregates the results according the aggregation method.

        Parameters
        ----------
        events
            A catalog or Event object
        waveforms
            Any object from which waveforms can be extracted.
        """
        waveforms = self._waveform_client if waveforms is None else waveforms
        df = self.calc_station_source_parameters(events=events, waveforms=waveforms,)
        out = SourceParameterAggregator()(df)
        return out

    def calc_station_source_parameters(
        self,
        events: Union[Event, Catalog],
        waveforms: Optional[Union[WaveformClient, Stream]] = None,
    ) -> pd.DataFrame:
        """
        Calculate source parameters by station.
        """
        waveforms = self._waveform_client if waveforms is None else waveforms

        stats_group = mopy.StatsGroup(
            inventory=self._inventory, catalog=events, restrict_to_arrivals=False
        )
        # Set any parameters that need setting on the stats_group
        for param in ["source_velocity", "quality_factor", "density"]:
            val = getattr(self, f"_{param}")
            if val is not None:
                set_param = getattr(stats_group, f"set_{param}")
                stats_group = set_param(val)

        trace_group = mopy.TraceGroup(
            stats_group=stats_group,
            waveforms=waveforms,
            motion_type="velocity",
            preprocess=self._stream_processor,
        ).fillna()

        # apply standard corrections on spectra
        spectrum_group = (
            trace_group.dft()
            .apply_default_corrections() 
            .ko_smooth()
            .dropna()
        )
        # then calculate source params
        sp = spectrum_group.calc_source_params()
        return sp

    def _get_prefilt(self, stream):
        """Get prefilt list."""
        srs = {tr.stats.sampling_rate for tr in stream}
        max_sr, min_sr = max(srs), min(srs)
        prefilt_low = list(self.prefilt_low)
        prefilt_high = [max_sr * x for x in self.prefilt_high]
        return prefilt_low + prefilt_high

    def _remove_response(self, stream: Stream):
        """Use the inventory to remove the response."""
        prefilt = self._get_prefilt(stream)
        st_out = stream.remove_response(
            inventory=self._inventory, pre_filt=prefilt,
        ).detrend("linear")
        return st_out
