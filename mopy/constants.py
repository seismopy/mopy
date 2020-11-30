"""
Constant values
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Union, Dict, Tuple, TypeVar

import pandas as pd

# from obsplus.constants import NSLC
from obspy import UTCDateTime
from obspy.core.event import Pick

# quality_factor = 400
# p_velocity = 3000
# s_velocity = p_velocity * 0.6
# s_rad = 0.60
# p_rad = 0.44
# density = 3000  # km/m**3

MOTION_TYPES = ("displacement", "velocity", "acceleration")

# The default body wave velocities
S_VELOCITY = 2400
P_VELOCITY = 4000

# The default radiation coefficients
S_RADIATION_COEFFICIENT = 0.6
P_RADIATION_COEFFICIENT = 0.44

# The quality factor for freq. dependent attenuation
QUALITY_FACTOR = 2000

# The Density in kg/m^3
DENSITY = 2700

# If the program is set to debug
DEBUG = False

# The default shear modulus (in Pa)
SHEAR_MODULUS = 2_200_000_000

# Minimum number of seconds per meter for phase windowing
SECONDS_PER_METER = 0.00003

# Minimum number of samples per phase window
MIN_SAMPLES = 60

# percentage of window added before and after phase window to taper
PERCENT_TAPER = 0.10

# The time the noise window ends before the P pick in seconds
NOISE_END_BEFORE_P = 1.0

# The minimum duration of the noise window
NOISE_MIN_DURATION = 1.0

# Expected columns/dtypes in the StatsGroup dataframe (should consider using something immutable?)
NSLC_DTYPES = OrderedDict(network=str, station=str, location=str, channel=str)

PICK_DTYPES = OrderedDict(
    time="datetime64[ns]",
    onset=str,
    polarity=str,
    method_id=str,
    pick_id=str,
    event_time="datetime64[ns]",
)

ARRIVAL_DTYPES = OrderedDict(distance_m="float64", azimuth="float64")

AMP_DTYPES = OrderedDict(starttime="datetime64[ns]", endtime="datetime64[ns]")

PHASE_WINDOW_INTERMEDIATES = OrderedDict(seed_id_less=str, phase_hint=str)
PHASE_WINDOW_INTERMEDIATE_DTYPES = PICK_DTYPES.copy()
PHASE_WINDOW_INTERMEDIATE_DTYPES.update(AMP_DTYPES)
PHASE_WINDOW_INTERMEDIATE_DTYPES.update(PHASE_WINDOW_INTERMEDIATES)

PHASE_WINDOW_DF_DTYPES = NSLC_DTYPES.copy()
PHASE_WINDOW_DF_DTYPES.update(PHASE_WINDOW_INTERMEDIATE_DTYPES)
PHASE_WINDOW_DF_DTYPES["seed_id"] = str

MOPY_SPECIFIC_DTYPES = OrderedDict(
    velocity="float64",
    radiation_coefficient="float64",
    quality_factor="float64",
    spreading_coefficient="float64",
    density="float64",
    shear_modulus="float64",
    free_surface_coefficient="float64",
)

# This currently only gets called by a test, which is a little odd?
STAT_DTYPES = NSLC_DTYPES.copy()
STAT_DTYPES.update(PICK_DTYPES)
STAT_DTYPES.update(ARRIVAL_DTYPES)
STAT_DTYPES.update(AMP_DTYPES)
STAT_DTYPES.update(MOPY_SPECIFIC_DTYPES)
STAT_DTYPES["sampling_rate"] = "float64"
STAT_DTYPES["vertical_distance_m"] = "float64"
STAT_DTYPES["hyp_distance_m"] = "float64"
STAT_DTYPES["ray_path_length_m"] = "float64"

DIST_COLS = (
    "ray_path_length_m",
    "hyp_distance_m",
    "azimuth",
    "vertical_distance_m",
    "distance_m",
)

_INDEX_NAMES = ("phase_hint", "event_id", "seed_id_less", "seed_id")


# ------- Type Hints (typically camel case)


# a generic type variable
Type1 = TypeVar("Type1")

# Types accepted for channel picks
ChannelPickType = Union[
    str, pd.DataFrame, Dict[Tuple[str, str, str], Union[UTCDateTime, Pick]]
]

# Types accepted for specifying absolute time windows
AbsoluteTimeWindowType = Union[
    str, pd.DataFrame, Dict[Tuple[str, str, str], Tuple[UTCDateTime, UTCDateTime]]
]

# Types for correction application
BroadcastableFloatType = Union[float, pd.Series]
