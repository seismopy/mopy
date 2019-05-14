"""
Constant values
"""
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

# Expected columns in the ChannelInfo dataframe
CHAN_COLS = (
    "network",
    "station",
    "location",
    "channel",
    "time",
    "tw_end",
    "tw_start",
    "sampling_rate",
    "distance",
    "horizontal_distance",
    "depth_distance",
    "azimuth",
    "ray_path_length",
    "velocity",
    "radiation_coefficient",
    "quality_factor",
    "spreading_coefficient",
    "density",
    "shear_modulus",
)
