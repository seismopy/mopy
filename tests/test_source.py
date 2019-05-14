"""
Tests for getting source params.
"""


import pandas as pd
import pytest


class TestFitModel:
    """ Tests for source params based on fitting a model to data. """

    # @pytest.fixture
    # def fitted_group(self, source_group_node):
    #     """ Fit a model to the group after smoothing and masking noise. """
    #     sg = source_group_node.abs().ko_smooth()
    #     # breakpoint()
    #     return sg.fit_source_model("brune")
    #
    # def test_type(self, fitted_group):
    #     """ ensure a dataframe was returned. """
    #     assert fitted_group.fit_df is not None
    #     assert isinstance(fitted_group.fit_df, pd.DataFrame)
    #
    #     breakpoint()
    #     fitted_group.plot(0, 0)
    #     assert isinstance(fitted_group, pd.DataFrame)
