# coding: utf-8

"""Code to visualize KV results.

"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import statsmodels.api as sm

from patsy import dmatrices
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.font_manager import FontProperties


# Upper limit of axes with predicted probability
PRED_MAX = 0.8

# Percentile to trim at borders.
TRIM_PCT = 5.0

# Small font for legend.
small_font = FontProperties()
small_font.set_size('small')


def get_lim(my_min, my_max):
    extra_space = (my_max - my_min) * 0.025
    return my_min - extra_space, my_max + extra_space


def get_grid(xmin, xmax, ymin, ymax):
    X, Y = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
    return X, Y


def plot_3d(model, trim_pct, elev=15, azim=70, flip_index_2=False, flip_index_1=False):

    model_index = model.index_final[0].values
    sdp_index = model.index_final[1].values

    mod_min = np.percentile(model_index, trim_pct)
    mod_max = np.percentile(model_index, 100 - trim_pct)
    sdp_min = np.percentile(sdp_index, trim_pct)
    sdp_max = np.percentile(sdp_index, 100 - trim_pct)

    sdp, mod = get_grid(sdp_min, sdp_max, mod_min, mod_max)

    eval_grid = np.array([np.concatenate(mod), np.concatenate(sdp)]).T
    pred_vec = model.semiparametric_probability_function(
        index=model.index_final,
        eval_locs=eval_grid
    )
    pred = np.array(np.split(pred_vec, 50))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(
        sdp, mod, pred,
        rstride=1,
        cstride=1,
        cmap=cm.coolwarm,
        norm=colors.Normalize(clip=False),
        linewidth=0,
        antialiased=False
    )

    ax.set_xlabel(labels["index_2"])
    ax.set_ylabel(labels["index_1"])
    ax.set_zlabel(labels["y"])

    # ax.view_init(elev=elev, azim=azim)

    ax.set_xlim(get_lim(sdp_min, sdp_max))
    ax.set_ylim(get_lim(mod_min, mod_max))

    ax.set_zlim(0.0, PRED_MAX)
    ax.zaxis.set_major_locator(LinearLocator(numticks=int(PRED_MAX * 10) + 1))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))

    if flip_index_2 is True:
        ax.invert_xaxis()
    if flip_index_1 is True:
        ax.invert_yaxis()

    plt.tight_layout()
    return fig


def plot_slices(model, trim_pct, slice_pctls, slice_index=0):

    xmin = np.percentile(model.index_final[slice_index], trim_pct)
    xmax = np.percentile(model.index_final[slice_index], 100 - trim_pct)

    model_grid = np.linspace(xmin, xmax, 50)

    fig = plt.figure()
    ax = plt.subplot(111)

    for q in slice_pctls:
        eval_grid = np.array(np.vstack(np.meshgrid(
            model_grid,
            np.percentile(model.index_final[slice_index], q)
        ))).T
        predict = model.semiparametric_probability_function(
            index=model.index_final,
            eval_locs=eval_grid
        )

        if q < 50:
            plt.plot(
                eval_grid[:, 0],
                predict,
                'b',
                ls='--',
                linewidth=2,
                label=r"{} at {}".format(
                    labels["index_{}".format(slice_index + 1)], q) + r"$^\mathrm{th}$ percentile"
            )
        else:
            plt.plot(
                eval_grid[:, 0],
                predict,
                'b',
                ls='-',
                linewidth=2,
                label=r"{} at {}".format(
                    labels["index_{}".format(slice_index + 1)], q) + r"$^\mathrm{th}$ percentile"
            )

    plt.xlabel(labels["index_{}".format(slice_index + 1)])
    plt.xlim(get_lim(xmin, xmax))

    plt.ylabel(labels['y'])
    plt.ylim((0, PRED_MAX))
    ax.yaxis.set_major_locator(LinearLocator(numticks=int(PRED_MAX * 10) + 1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.01f'))

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([
        box.x0,
        box.y0 + box.height * 0.15,
        box.width,
        box.height * 0.85
    ])

    # Put a legend below current axis
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        fancybox=True,
        shadow=False,
        ncol=2,
        prop=small_font
    )

    return fig


def plot_dens_contour(model, trim_pct):

    model_index = model.index_final[0].values
    sdp_index = model.index_final[1].values

    xmin = np.percentile(model_index, trim_pct)
    xmax = np.percentile(model_index, 100 - trim_pct)
    ymin = np.percentile(sdp_index, trim_pct)
    ymax = np.percentile(sdp_index, 100 - trim_pct)

    X, Y = get_grid(xmin, xmax, ymin, ymax)
    eval_grid = np.array([np.concatenate(X), np.concatenate(Y)]).T

    Zvec = model.f(eval_grid=eval_grid, index_data=model.index_final.values)
    Z = np.array(np.split(Zvec, 50))

    # Normalise density
    # Z *= np.linalg.det(np.cov(model.index_final.values.T)) ** 0.5

    fig = plt.figure()
    ct = plt.contour(X, Y, Z, colors='b')
    plt.clabel(ct, inline=1, fontsize=8, fmt='%1.3g')

    plt.xlabel(labels["index_2"])
    plt.ylabel(labels["index_1"])

    # plt.xlim(get_lim(xmin, xmax))
    # plt.ylim(get_lim(ymin, ymax))

    return fig


def plot_asf(model, trim_pct, asf_index_loc=0, r=None, ε=1e-3):

    xmin = np.percentile(model.index_final[asf_index_loc], trim_pct)
    xmax = np.percentile(model.index_final[asf_index_loc], 100 - trim_pct)
    grid = np.linspace(xmin, xmax, 20)

    asf = np.empty(grid.shape) * np.nan
    asf_se = np.empty(grid.shape)
    for i, g in enumerate(grid):
        asf[i], asf_se[i] = model.average_structural_function(
            asf_index_loc=asf_index_loc,
            asf_loc=g,
            r=None, ε=1e-3
        )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plt.plot(grid, asf)
    plt.plot(grid, asf + 1.96 * asf_se, 'b--')
    plt.plot(grid, asf - 1.96 * asf_se, 'b--')

    if asf_index_loc == 0:
        plt.xlabel(labels["index_1"])
    elif asf_index_loc == 1:
        plt.xlabel(labels["index_2"])

    plt.ylabel(labels["y"])
    plt.ylim((0, PRED_MAX))
    ax.yaxis.set_major_locator(LinearLocator(numticks=int(PRED_MAX * 10) + 1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.01f'))
    plt.xlim(get_lim(xmin, xmax))

    return fig


def plot_probit(model, trim_pct, probit_index_loc=0):
    '''Return plot of probit for specification used in *model*, use full dataset.

    '''

    data = pd.read_csv('./data/' + model_json['data'])
    exog_vars = ' + '.join(model.coeffs_final[probit_index_loc].index)

    Y, X = dmatrices(model_json['y_name'] + ' ~ ' + exog_vars, data)
    probit_result = sm.Probit(Y, X).fit()

    Xb = np.sort(probit_result.fittedvalues)
    p_hat = np.sort(probit_result.predict())

    # To make plots comparable, standardize Xb, rescale to mean and variance of index.
    μ_Xb = Xb.mean()
    σ_Xb = Xb.std()
    Xb_standardized = (Xb - μ_Xb) / σ_Xb

    index = model.index_final[probit_index_loc]
    μ_index = index.mean()
    σ_index = index.std()

    Xb_scaled = Xb_standardized * σ_index + μ_index

    # Align limits with ASF limits.
    xmin = np.percentile(model.index_final[probit_index_loc], trim_pct)
    xmax = np.percentile(model.index_final[probit_index_loc], 100 - trim_pct)

    fig = plt.figure()
    fig.add_subplot(1, 1, 1)

    plt.xlim(get_lim(xmin, xmax))
    plt.ylim((0, PRED_MAX))

    if probit_index_loc == 0:
        plt.xlabel('Rescaled probit index 1')
    elif probit_index_loc == 1:
        plt.xlabel('Rescaled probit index 2')

    plt.ylabel(labels["y"])
    plt.plot(Xb_scaled, p_hat)

    return fig


def plot_asf_and_probit(model, trim_pct, asf_index_loc=0, r=None, ε=1e-3):

    xmin = np.percentile(model.index_final[asf_index_loc], trim_pct)
    xmax = np.percentile(model.index_final[asf_index_loc], 100 - trim_pct)
    grid = np.linspace(xmin, xmax, 20)

    # ASF plot.
    asf = np.empty(grid.shape) * np.nan
    asf_se = np.empty(grid.shape)
    for i, g in enumerate(grid):
        asf[i], asf_se[i] = model.average_structural_function(
            asf_index_loc=asf_index_loc,
            asf_loc=g,
            r=None, ε=1e-3
        )

    fig = plt.figure()
    ax = plt.subplot(111)

    plt.plot(
        grid,
        asf,
        label='Average structural function',
        color='b',
        ls='-',
        linewidth=2
    )

    # Probit plot.
    data = pd.read_csv('./data/' + model_json['data'])
    exog_vars = ' + '.join(model.coeffs_final[asf_index_loc].index)

    Y, X = dmatrices(model_json['y_name'] + ' ~ ' + exog_vars, data)
    probit_result = sm.Probit(Y, X).fit()

    Xb = np.sort(probit_result.fittedvalues)
    p_hat = np.sort(probit_result.predict())

    # To make plots comparable, standardize Xb, rescale to mean and variance of index.
    μ_Xb = Xb.mean()
    σ_Xb = Xb.std()
    Xb_standardized = (Xb - μ_Xb) / σ_Xb

    index = model.index_final[asf_index_loc]
    μ_index = index.mean()
    σ_index = index.std()

    Xb_scaled = Xb_standardized * σ_index + μ_index

    # Align limits with ASF limits.
    xmin = np.percentile(model.index_final[asf_index_loc], trim_pct)
    xmax = np.percentile(model.index_final[asf_index_loc], 100 - trim_pct)

    plt.plot(
        Xb_scaled,
        p_hat,
        label='Probit prediction',
        color='b',
        ls='--',
        linewidth=2
    )

    # Plot.
    if asf_index_loc == 0:
        plt.xlabel('Rescaled probit index 1')
    elif asf_index_loc == 1:
        plt.xlabel('Rescaled probit index 2')

    plt.ylabel(labels['y'])
    plt.ylim((0, PRED_MAX))
    ax.yaxis.set_major_locator(LinearLocator(numticks=int(PRED_MAX * 10) + 1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.01f'))
    plt.xlim(get_lim(xmin, xmax))

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([
        box.x0,
        box.y0 + box.height * 0.15,
        box.width,
        box.height * 0.85
    ])

    # Put a legend below current axis
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        fancybox=True,
        shadow=False,
        ncol=2,
        prop=small_font
    )

    return fig


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Provide model name and figure type as arguments."
    model_name = sys.argv[1]
    figure_type = sys.argv[2]

    # Create directory for output.
    if not os.path.exists('./bld/figures'):
        os.makedirs('./bld/figures')

    model_json = json.load(open(
        './models/{}.json'.format(model_name), encoding='utf-8'))
    labels = model_json["labels"]

    with open('./bld/fitted/kv_{}.pickle'.format(model_name), mode='rb') as f:
        kv_model = pickle.load(f)

    if figure_type in ['3d', 'all']:
        fig = plot_3d(model=kv_model, trim_pct=TRIM_PCT, flip_index_2=False, flip_index_1=False)
        fig.savefig('./bld/figures/kv_{}_3d.pdf'.format(model_name), dpi=400)
    if figure_type in ['dens_contour', 'all']:
        fig = plot_dens_contour(model=kv_model, trim_pct=TRIM_PCT)
        fig.savefig('./bld/figures/kv_{}_dens_contour.pdf'.format(model_name), dpi=400)
    if figure_type in ['slices_index_1', 'all']:
        fig = plot_slices(model=kv_model, trim_pct=TRIM_PCT, slice_pctls=[10, 90], slice_index=0)
        fig.savefig('./bld/figures/kv_{}_slices_index_1.pdf'.format(model_name), dpi=400)
    if figure_type in ['slices_index_2', 'all']:
        fig = plot_slices(model=kv_model, trim_pct=TRIM_PCT, slice_pctls=[10, 90], slice_index=1)
        fig.savefig('./bld/figures/kv_{}_slices_index_2.pdf'.format(model_name), dpi=400)
    if figure_type in ['asf_index_1', 'all']:
        fig = plot_asf(model=kv_model, trim_pct=TRIM_PCT, asf_index_loc=0)
        fig.savefig('./bld/figures/kv_{}_asf_index_1.pdf'.format(model_name), dpi=400)
    if figure_type in ['asf_index_2', 'all']:
        fig = plot_asf(model=kv_model, trim_pct=TRIM_PCT, asf_index_loc=1)
        fig.savefig('./bld/figures/kv_{}_asf_index_2.pdf'.format(model_name), dpi=400)
    if figure_type in ['probit_index_1', 'all']:
        fig = plot_probit(model=kv_model, trim_pct=TRIM_PCT, probit_index_loc=0)
        fig.savefig('./bld/figures/kv_{}_probit_index_1.pdf'.format(model_name), dpi=400)
    if figure_type in ['probit_index_2', 'all']:
        fig = plot_probit(model=kv_model, trim_pct=TRIM_PCT, probit_index_loc=1)
        fig.savefig('./bld/figures/kv_{}_probit_index_2.pdf'.format(model_name), dpi=400)
    if figure_type in ['asf_and_probit_index_1', 'all']:
        fig = plot_asf_and_probit(model=kv_model, trim_pct=TRIM_PCT, asf_index_loc=0)
        fig.savefig('./bld/figures/kv_{}_asf_and_probit_index_1.pdf'.format(model_name), dpi=400)
    if figure_type in ['asf_and_probit_index_2', 'all']:
        fig = plot_asf_and_probit(model=kv_model, trim_pct=TRIM_PCT, asf_index_loc=1)
        fig.savefig('./bld/figures/kv_{}_asf_and_probit_index_2.pdf'.format(model_name), dpi=400)

    if figure_type not in [
        'all',
        '3d',
        'slices_index_1',
        'slices_index_2',
        'dens_contour',
        'asf_index_1',
        'asf_index_2',
        'probit_index_1',
        'probit_index_2',
        'asf_and_probit_index_1',
        'asf_and_probit_index_2'
    ]:
        raise ValueError(figure_type)
