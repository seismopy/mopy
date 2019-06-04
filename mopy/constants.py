"""
Constant values
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Union, Dict, Tuple, TypeVar

import pandas as pd
from obsplus.constants import NSLC
from obspy import UTCDateTime
from obspy.core.event import Pick

quality_factor = 400
p_velocity = 3000
s_velocity = p_velocity * 0.6
s_rad = 0.60
p_rad = 0.44
density = 3000  # km/m**3

MOTION_TYPES = ("displacement", "velocity", "acceleration")

# The default body wave velocities
S_VELOCITY = 2_400
P_VELOCITY = 4_000

# The default radiation coefficients
S_RADIATION_COEFFICIENT = 0.6
P_RADIATION_COEFFICIENT = 0.44

# The quality factor for freq. dependent attenuation
QUALITY_FACTOR = 2000

# The Density in kg/m^3
DESNITY = 2700

# If the program is set to debug
DEBUG = False

# The default shear modulus (in Pa)
SHEAR_MODULUS = 2_200_000_000

# Minimum number of seconds per meter for phase windowing
SECONDS_PER_METER = 0.000_03

# Minimum number of samples per phase window
MIN_SAMPLES = 60

# percentage of window added before and after phase window to taper
PERCENT_TAPER = 0.10

# The time the noise window ends before the P pick in seconds
NOISE_END_BEFORE_P = 1.0

# The minimum duration of the noise window
NOISE_MIN_DURATION = 1.0

# Expected columns in the ChannelInfo dataframe
PICK_COLS = ("time", "onset", "polarity", "method_id", "pick_id")

ARRIVAL_COLS = ("distance", "azimuth")

AMP_COLS = ("starttime", "endtime")

CHAN_COLS = (
    NSLC
    + PICK_COLS
    + ARRIVAL_COLS
    + AMP_COLS
    + (
        "sampling_rate",
        "horizontal_distance",
        "depth_distance",
        "ray_path_length",
        "velocity",
        "radiation_coefficient",
        "quality_factor",
        "spreading_coefficient",
        "density",
        "shear_modulus",
    )
)

# Datatypes for columns in the ChannelInfo dataframe
# ^ not quite sure how or when to force this... cannot specify by column during instantiation, irritatingly
CHAN_DTYPES = OrderedDict(
    network=str,
    station=str,
    location=str,
    channel=str,
    time=float,
    starttime=float,
    endtime=float,
    sampling_rate=float,
    distance=float,
    horizontal_distance=float,
    depth_distance=float,
    azimuth=float,
    ray_path_length=float,
    velocity=float,
    radiation_coefficient=float,
    quality_factor=int,
    spreading_coefficient=float,
    density=float,
    shear_modulus=float,
    onset=str,
    polarity=str,
    method_id=str,
    pick_id=str,
)

_INDEX_NAMES = ("phase_hint", "event_id", "seed_id")

# ------- Type Hints (typically camel case)


# a generic type variable
Type1 = TypeVar("Type1")

# Types accepted for channel picks
ChannelPickType = Union[
    str, pd.DataFrame, Dict[Tuple[str, str, str], Union[UTCDateTime, Pick]]
]
