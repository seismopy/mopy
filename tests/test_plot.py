"""
Tests for plotting.
"""


class TestPlotSourceGroup:
    """
    Tests for plotting the source groups. These don't do any comparisons of
    images; they simply ensure the image generation code can run.
    """

    def test_plot_event_spectra(self, spectrum_group_node):
        """ Ensure sources can be plotted. """
        norm = spectrum_group_node.abs().ko_smooth()
        # this is to make sure the code can run, doesn't compare images
        norm.plot(0, show=False, limit=5)
        # make sure limit can be one
        norm.plot(0, show=False, limit=1)

    # def test_plot_time_domain(self, source_group):
    #     """ tests for plotting the time domain rep of events."""
    #     source_group.plot_time_domain(0, show=False)
