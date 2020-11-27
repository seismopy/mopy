#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various FFT helper functions

Created on 11/23/20

@author: SBoltz
"""

import warnings

import pandas as pd


def dft_to_cft(df: pd.DataFrame, sampling_rate: float, **kwargs):
    return df.copy().divide(sampling_rate, axis=0)


def cft_to_dft(df: pd.DataFrame, sampling_rate: float, **kwargs):
    return df.copy().multiply(sampling_rate, axis=0)


def dft_to_psd(df: pd.DataFrame, sampling_rate: float, npoints: pd.Series, **kwargs):
    df = df.copy()
    fft_sq = df.pow(2)
    fft_normed = fft_sq.divide(sampling_rate*npoints, axis=0)
    # double the non-zero components to account for negative frequencies
    fft_normed.iloc[:, 1:] *= 2
    return fft_normed


def psd_to_dft(df: pd.DataFrame, sampling_rate: float, npoints: pd.Series, **kwargs):
    warnings.warn("Converting from 'psd' will result in the loss of sign information")  # TODO: This error message is a little bit misleading? It's technically converting to the PSD that loses the info, but need some way to emphasize that you're not going to get the original DFT back here
    df = df.copy()
    df.iloc[:, 1:] /= 2
    df_denormed = df.multiply(sampling_rate*npoints,  axis=0)
    return df_denormed.pow(1/2)


def cft_to_psd(df: pd.DataFrame, sampling_rate: float, npoints: pd.Series, **kwargs):
    dft = cft_to_dft(df, sampling_rate=sampling_rate)
    return dft_to_psd(dft, sampling_rate=sampling_rate, npoints=npoints)


def psd_to_cft(df: pd.DataFrame, sampling_rate: float, npoints: pd.Series, **kwargs):
    dft = psd_to_dft(df, sampling_rate=sampling_rate, npoints=npoints)
    return dft_to_cft(dft, sampling_rate=sampling_rate)
