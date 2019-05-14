"""
Module for source models and fitting.
"""
import warnings
from functools import partial

import mopy
import numpy as np
import pandas as pd
import scipy.optimize
from mopy.constants import MOTION_TYPES

# Define parameters that make source models unique
SOURCE_MODEL_PARAMS = {
    "brune": dict(n=2, gamma=1),
    "boatwright": dict(n=2, gamma=2),
    "haskell": dict(n=3, gamma=2.0 / 3.0),
}

INVERSION_PARAMS = ("omega0", "fc", "quality_factor")


# define dict to allow fitting sources to displacement velocity or acceleration
def _get_modification_factor(f: np.array, ground_motion_type: str):
    """
    Get the modification to apply to brune(ish) models if non-displacement
    is provided.
    """
    assert ground_motion_type in MOTION_TYPES
    if ground_motion_type == "displacement":
        return 1
    elif ground_motion_type == "velocity":
        return f * 2 * np.pi
    elif ground_motion_type == "acceleration":
        return (f * 2 * np.pi) ** 2


def source_spectrum(
    freqs: np.array,
    omega0: float,
    fc: float,
    quality_mod: float = 1,
    gamma: float = 1,
    n: float = 2,
    motion_type="displacement",
) -> np.array:
    """
    Return a theoretical source spectrum for input frequencies f
    """
    # set bounds
    if fc < 0 or fc >= freqs.max():
        return [-9999] * len(freqs)
    mod_fact = _get_modification_factor(freqs, ground_motion_type=motion_type)
    denom = (1 + (freqs / fc) ** (gamma * n)) ** (1 / gamma)
    out = np.abs((omega0 * mod_fact * quality_mod) / denom)
    return out


def _optimize(x_0, y_0, p0, pbounds, motion_type):
    """ """
    breakpoint()
    # scale variables to min and max

    omega = np.log(p0)
    fc = p0[1]

    def _fun_to_opt(p0):

        out = source_spectrum(x_0, *p0, motion_type=motion_type)

        # define the cost function
        cost = (out - y_0) ** 2 / (np.linalg.norm(out) * np.linalg.norm(y_0))

    scipy.optimize.minimize(_fun_to_opt, np.ndarray([omega, fc]), method="nelder-mead")


def _get_starting_and_bounds(df, stats) -> pd.DataFrame:
    """
    Return a dataframe with starting values and sensible limits of source params.
    """
    # assert stats.motion_type == "velocity", "only velocity for now"
    min_index = 3  # the starting index to allow param determination (see below)
    # trim the dataframe to only include data starting at 4th lowest freq
    # this is just used to get reasonable starting values not too close to 0
    dfs = df[df.columns[min_index:]]
    # get estimate of omega0 by integrating
    mod = _get_modification_factor(dfs.columns, stats.motion_type)
    omega0 = (dfs / mod).max(axis=1)
    out = dict(fc=dfs.idxmax(axis=1), omega0=omega0, quality_factor=100)
    # only allow omega to vary 2 orders of mag in either direction
    out["omega0_max"] = out["omega0"] * 100
    out["omega0_min"] = out["omega0"] / 100
    # set fc_max at nyquist and fc_min and lowest freq allows by min_index
    out["fc_max"] = stats.sampling_rate / 2.0
    out["fc_min"] = dfs.columns[0]
    # set q limits
    out["quality_factor_min"] = 5
    out["quality_factor_max"] = 1000
    # get dataframe perform sanity check and return
    df_out = pd.DataFrame(out)
    assert not df_out.isnull().any().any()
    return df_out


def _fit_row(row, opt_func, param_names, params_df, method, raise_on_fail, **kwargs):
    """ Apply fit to a row of the dataframe (contains spectrum). """

    # filter out any null values
    # get parameters and bounds/initial values for this index
    params = params_df.loc[row.name]
    bounds = np.array(
        [
            tuple(params[f"{x}_min"] for x in param_names),
            tuple(params[f"{x}_max"] for x in param_names),
        ]
    )
    p0 = params[list(param_names)].values

    is_null = row.isnull()
    # init kwargs for curve fitting
    x_data = row.index.values[~is_null]
    y_data = row.values[~is_null]

    breakpoint()

    _optimize(x_data, y_data, p0, bounds, motion_type="velocity")

    kwargs.update(
        dict(
            f=opt_func, xdata=x_data, ydata=y_data, p0=p0, bounds=bounds, method=method
        )
    )
    # curve fit
    try:
        popt, pcov = scipy.optimize.curve_fit(**kwargs)
        if (popt == p0).all():
            msg = (
                f"for {row.name} initial values of {p0} were identical to "
                f"optimized values of {popt}, failing optimization."
            )
            raise RuntimeError(msg)
    except RuntimeError as e:
        msg = f"failed to fit source  to {row.name}"
        if e.args:
            msg += f" returned message: {e.args[0]}"
        if raise_on_fail:
            raise RuntimeError(msg)
        else:
            warnings.warn(msg)
    else:
        # add results to results dict
        res_dict = {name: val for name, val in zip(param_names, popt)}
        # and error estimates to results dict
        stds = np.diag(pcov)
        for name, std in zip(param_names, stds):
            res_dict[f"{name}_error"] = std

        import matplotlib.pyplot as plt

        fit_0 = opt_func(x_data, *p0)
        fit_opt = opt_func(x_data, *popt)
        fit_me = opt_func(x_data, popt[0], 15)

        plt.loglog(x_data, fit_0, "r")
        plt.loglog(x_data, y_data, "b")
        plt.loglog(x_data, fit_opt, "g")
        plt.loglog(x_data, fit_me, "k")

        plt.show()
        breakpoint()

        return pd.Series(res_dict)


def fit_model(
    source_group: "mopy.SourceGroup",
    model="brune",
    raise_on_fail=False,
    invert_q=False,
    method="dogbox",
    fit_noise=False,
) -> pd.DataFrame:
    """
    Use scipy to fit model to spectrum.
    """
    assert model in SOURCE_MODEL_PARAMS, f"{model} is not a supported model."
    source_kwargs = SOURCE_MODEL_PARAMS[model]

    motion_type = source_group.stats.motion_type

    func = partial(source_spectrum, motion_type=motion_type, **source_kwargs)
    param_names = INVERSION_PARAMS if invert_q else INVERSION_PARAMS[:2]
    # get dataframe, filter out any rows with all NaNs
    df = source_group.data
    df = df[~df.isnull().all(axis=1)]
    # filter out noise if fit_noise is false
    if not fit_noise:
        df = df.loc[df.index.get_level_values("phase_hint") != "Noise"]
    # get starting values and sensible bounds
    params_df = _get_starting_and_bounds(df, source_group.stats)

    out = df.apply(
        _fit_row,
        axis=1,
        opt_func=func,
        param_names=param_names,
        params_df=params_df,
        method=method,
        raise_on_fail=raise_on_fail,
    )
    # add source model info
    assert isinstance(out, pd.DataFrame)
    out.columns = pd.MultiIndex.from_product(
        [[model], out.columns], names=["model", "parameters"]
    )
    return out
