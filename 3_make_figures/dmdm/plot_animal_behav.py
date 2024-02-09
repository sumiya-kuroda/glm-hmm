from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sb
import pandas as pd


def plot_PC(df: pd.DataFrame, ax, label, K=None):
    """
    Plot psychometric curve
    i.e. hit / (hit + miss)
    """
    if K is None:
        hitlicks = (df[~df['early_report'] & (df['fitted_trials'] == 1)]
                    .groupby(['sig', 'session'])
                    .agg({'hit': 'mean'})
                    .unstack()
                    )
    else:
        hitlicks = (df[~df['early_report'] & (df['fitted_trials'] == 1) & (df['state'] == K)]
                    .groupby(['sig', 'session'])
                    .agg({'hit': 'mean'})
                    .unstack()
                    )

    hitlicks_mean = hitlicks.mean(axis=1)
    hitlicks_prc = hitlicks.quantile([0.025, 0.975], axis=1)

    ax.plot(hitlicks_mean, color='k', label=label, alpha=0.8)
    ax.fill_between(
        hitlicks.index, hitlicks_prc.loc[0.025], hitlicks_prc.loc[0.975],
        alpha=0.3, lw=0, color='k'
    )

def plot_CC(df: pd.DataFrame, ax, label, K=None):
    """
    Plot chronometric curve
    """
    if K is None:
        hitrt = (
                df[df['hit'] & (df['sig'] > 0) & (df['fitted_trials'] == 1)]
                .groupby(['sig', 'session'])
                .agg({'rt_change': 'median'})
                .unstack()
                )
    else:
        hitrt = (
                df[df['hit'] & (df['sig'] > 0) & (df['fitted_trials'] == 1) & (df['state'] == K)]
                .groupby(['sig', 'session'])
                .agg({'rt_change': 'median'})
                .unstack()
                )
        
    hitrt_mean = hitrt.median(axis=1)
    hitrt_prc = hitrt.quantile([0.025, 0.975], axis=1)

    ax.plot(hitrt_mean, color='k', alpha=0.8)
    ax.fill_between(
        hitrt.index, hitrt_prc.loc[0.025], hitrt_prc.loc[0.975],
        alpha=0.3, lw=0, color='k'
    )


def plot_FArate(df: pd.DataFrame, ax, label, K=None):
    """
    Plot false alarm rate (early lick/press rate)
    """

    if K is None:
        earlylicks = (df[(df['fitted_trials'] == 1)]
                    .groupby(['session'])
                    .agg({'early_report': 'mean'})
                    .unstack()
                    )
    else:
        earlylicks = (df[(df['fitted_trials'] == 1) & (df['state'] == K)]
                    .groupby(['session'])
                    .agg({'early_report': 'mean'})
                    .unstack()
                    )
        
    earlylicks_mean = earlylicks.mean(axis=0)
    if earlylicks.dtype == 'float64':
        earlylicks_prc = earlylicks.quantile([0.025, 0.975]) # this gives actual values

        earlylicks_err = np.array([[earlylicks_mean - earlylicks_prc.values[0]], 
                                [earlylicks_prc.values[1] - earlylicks_mean]])

        ax.errorbar(0, earlylicks_mean, yerr=earlylicks_err, 
                    color='k', marker='o',
                    markersize='2', capsize=2) # this asks for the diff

# def plot_HazardRate(df: pd.DataFrame, ax, label, K=None):
#     """
#     Plot false alarm rate (early lick/press rate)
#     """

#     if K is None:
#         earlylicks = (df[(df['fitted_trials'] == 1)]
#                     .groupby(['session'])
#                     .agg({'early_report': 'mean'})
#                     .unstack()
#                     )
#     else:
#         earlylicks = (df[(df['fitted_trials'] == 1) & (df['state'] == K)]
#                     .groupby(['session'])
#                     .agg({'early_report': 'mean'})
#                     .unstack()
#                     )
        
#     earlylicks_mean = earlylicks.mean(axis=0)
#     earlylicks_prc = earlylicks.quantile([0.025, 0.975]) # this gives actual values

#     earlylicks_err = np.array([[earlylicks_mean - earlylicks_prc.values[0]], 
#                                [earlylicks_prc.values[1] - earlylicks_mean]])

#     ax.errorbar(0, earlylicks_mean, yerr=earlylicks_err, 
#                 color='k', marker='o') # this asks for the diff
