"""
Core classes
"""

from mopy.core.channelinfo import ChannelInfo
from mopy.core.base import DataFrameGroupBase
from mopy.core.tracegroup import TraceGroup
from mopy.core.spectrumgroup import SpectrumGroup

# A map to get a function which returns the constant to multiply to perform
# temporal integration/division in the freq. domain

# This is type annotation to specify subclass outputs of parent methods on
# the DataFrameGroupBase type


# df[('maxmean', 'mw')] = (2/3) * np.log10(df[('maxmean', 'moment')]) - 6.0
# dff = df['maxmean']
# dff['potency'] = dff['moment'].divide(self.meta['shear_modulus'], axis=0)
# dff['apparent_stress'] = dff['energy'] / dff['potency']
# import matplotlib.pyplot as plt
#
#
# plt.plot(np.log10(dff['potency']), np.log10(dff['energy']), '.')
#
# plt.show()
#
# breakpoint()
#
# apparent = dff['apparent_stress'].dropna()
#
#
#
# apparent = apparent[~np.isinf(apparent)]
#
#
#
# # apparent.hist()
# # plt.show()
# breakpoint()
#
#
#
# # add data frame to results