"""
Relationship among self-reports of dream/wake pos/neg emotion.

Run repeated measures correlation, export stats table and visualization.
"""
import os
import json

import numpy as np
import pandas as pd

import pingouin as pg
from statsmodels.formula.api import ols

import seaborn as sea
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["savefig.dpi"] = 600
plt.rcParams["interactive"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams["mathtext.it"] = "Arial:italic"
plt.rcParams["mathtext.bf"] = "Arial:bold"


with open("./config.json","r") as config_file:
    P = json.load(config_file)
    DATA_DIRECTORY = P["data_directory"]
    RESULTS_DIRECTORY = P["results_directory"]
    COLORS = P["colors"]


IMPORT_BASENAME = "data_anon.csv"
EXPORT_BASENAME = "correlations"


import_filename = os.path.join(DATA_DIRECTORY, IMPORT_BASENAME)
export_stats_filename = os.path.join(RESULTS_DIRECTORY, f"{EXPORT_BASENAME}.csv")
export_plot_filename = os.path.join(RESULTS_DIRECTORY, f"{EXPORT_BASENAME}.png")

# load data
df = pd.read_csv(import_filename)

# run both repeated measures correlations
neg_corr = pg.rm_corr(data=df, x="dream_negemo", y="wake_negemo", subject="participant")
pos_corr = pg.rm_corr(data=df, x="dream_posemo", y="wake_posemo", subject="participant")
negpos_corr = pg.rm_corr(data=df, x="wake_negemo", y="wake_posemo", subject="participant")

# save out stats table
stats_df = pd.concat([neg_corr, pos_corr, negpos_corr])
stats_df = stats_df.rename_axis("test").reset_index()
stats_df.index = pd.Index(["negative", "positive", "wake_negpos"], name="valence")
stats_df.round(5).to_csv(export_stats_filename, index=True)



########## plot



# Each cell of the plot is a grid, and it should be square.
# This means there might be some empty squares in each grid.
# That's okay.
# EVERYTHING WILL BE OKAY.
# To find the grid dimensions, take square root of the number of
# subjects, then round up. Squaring that will be grid size (of each cell).
n_subjects = df["participant"].nunique()
grid_dim = int( np.ceil( np.sqrt( n_subjects ) ) )
n_xtra_cells = np.square(grid_dim) - n_subjects

neg_colors = [ "white", COLORS["negative"]]
pos_colors = [ "white", COLORS["positive"]]
neg_c = COLORS["negative"]
pos_c = COLORS["positive"]
COLORMAPS = {
    "neg" : LinearSegmentedColormap.from_list("mycmap", neg_colors),
    "pos" : LinearSegmentedColormap.from_list("mycmap", pos_colors),
}

# set figure and axis size stuff
FIGW = 3.1
FIGH = 5
# goal is to get the same size in inches or w/e for each subplot
AX_HEIGHT_FRACTION = .4 # of a single valence grid
h_inches = FIGH * AX_HEIGHT_FRACTION # of subplot
width_fraction = h_inches / FIGW
w_inches = FIGW * width_fraction # of subplot


fig = plt.figure(figsize=(FIGW,FIGH), constrained_layout=False)

SPACING = .1 # between the two valence axes

gs1_TOP = .95
gs1_BOTTOM = .55
gs1 = fig.add_gridspec(ncols=9, nrows=9, wspace=SPACING, hspace=SPACING,
    top=gs1_TOP, bottom=gs1_BOTTOM,
    left=0+((1-width_fraction)/2),
    right=1-((1-width_fraction)/2))

gs_cbar1 = fig.add_gridspec(top=gs1_TOP, bottom=gs1_BOTTOM,
    left=1-((1-width_fraction)/2) + .02,
    right=1-((1-width_fraction)/2) + .04)
cbar_ax1 = fig.add_subplot(gs_cbar1[:,:])

gs2_TOP = .51
gs2_BOTTOM = .11
gs2 = fig.add_gridspec(ncols=9, nrows=9, wspace=SPACING, hspace=SPACING,
    top=gs2_TOP, bottom=gs2_BOTTOM,
    left=0+((1-width_fraction)/2),
    right=1-((1-width_fraction)/2))

gs_cbar2 = fig.add_gridspec(top=gs2_TOP, bottom=gs2_BOTTOM,
    left=1-((1-width_fraction)/2) + .02,
    right=1-((1-width_fraction)/2) + .04)
cbar_ax2 = fig.add_subplot(gs_cbar2[:,:])


REGPLOT_ARGS = {
    "scatter" : False,
    "ci" : None,
    "truncate" : True,
    "line_kws" : dict(linewidth=.4, alpha=.4),
}
LINE_ARGS = { # for the grid
    "linewidth" : .5,
    "alpha" : .5,
    "color" : "gainsboro",
}


# to keep color normalization constant across both axes
# get the highest possible value here
max_neg = df.groupby(["participant","dream_negemo","wake_negemo"]).size().max()
max_pos = df.groupby(["participant","dream_posemo","wake_posemo"]).size().max()

colornorm = LogNorm(vmin=.1, vmax=max([max_neg,max_pos]))


for val, gs in zip(["neg", "pos"], [gs1, gs2]):
    # make an index full of all 1-9 combos to avoid missing stuff
    likert_combo_index = pd.MultiIndex.from_product([range(1,10),range(1,10)],
        names=[f"dream_{val}emo",f"wake_{val}emo"])

    # restructure so that each column is a subject
    # and each row is a dream/wake likert response combo
    gridlike_df = df.groupby([f"dream_{val}emo", f"wake_{val}emo", "participant"]
        ).size(
        ).unstack(
        ).reindex(likert_combo_index)

    cmap = COLORMAPS[val]
    color = COLORS["negative"] if val == "neg" else COLORS["positive"]

    for d in range(9):
        for w in range(9):
            valus = gridlike_df.loc[(d+1,w+1)].values
            valus = np.append(valus, np.full(n_xtra_cells, np.nan))
            mat = valus.reshape(grid_dim,grid_dim, order="C")
            ax = fig.add_subplot(gs[8-w,d])
            ax.patch.set_visible(False) # remove white background so regression lines come through
            im = ax.imshow(mat, cmap=cmap,
                origin="upper",
                extent=(0, grid_dim, 0, grid_dim), # (left, right, bottom, top)
                norm=colornorm,
            )

            # colorbar
            if val == "neg":
                cbar_ax = cbar_ax1
            else:
                cbar_ax = cbar_ax2
            cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
            cbar.ax.tick_params(which="major", color="white", size=3, direction="in", labelsize=8)
            cbar.ax.tick_params(which="minor", color="white", size=2, direction="in")
            cbar.outline.set_visible(False)
            # cbar.outline.set_linewidth(.5)
            cbar.ax.set_ybound(lower=1)
            if val == "neg":
                cbar.ax.yaxis.set_ticklabels([])
            if val == "pos":
                cbar.set_label("# of morning reports",
                    labelpad=6, rotation=270, fontsize=8,
                    va="center", ha="center")

            # make custom gridlines
            line_locs = np.arange(1, grid_dim)
            xmins = np.full(grid_dim-1, 0)
            xmaxs = np.full(grid_dim-1, grid_dim)
            ymins = np.full(grid_dim-1, 0)
            ymins[-(n_xtra_cells-1):] = 1
            ymaxs = np.full(grid_dim-1, grid_dim)
            ax.hlines(line_locs, xmins, xmaxs, **LINE_ARGS)
            ax.vlines(line_locs, ymins, ymaxs, **LINE_ARGS)

            # tick locators
            ax.xaxis.set(major_locator=plt.NullLocator())
            ax.yaxis.set(major_locator=plt.NullLocator())
            if ax.is_last_row():
                ax.set_xticks([grid_dim/2])
                ax.set_xticklabels([])
                if val == "pos":
                    ax.set_xticks([grid_dim/2])
                    ax.set_xticklabels([d+1])
                    if d+1 == 5:
                        ax.set_xlabel("dream emotion", fontsize=10)
            if ax.is_first_col():
                ax.set_yticks([grid_dim/2])
                ax.set_yticklabels([])
                if val == "pos":
                    ax.set_yticklabels([w+1])
                    if w+1 == 5:
                        ax.set_ylabel("wake emotion", fontsize=10)


    topax = fig.add_subplot(gs[:,:], frame_on=False)
    topax.set_zorder(-1)
    subj_col = "participant"
    iv_col = f"wake_{val}emo"
    dv_col = f"dream_{val}emo"
    data = df[[dv_col, iv_col, subj_col]].dropna(axis=0)
    ols_formula = f"Q('{iv_col}') ~ C(Q('{subj_col}')) + Q('{dv_col}')"
    model = ols(ols_formula, data=data).fit()
    data["pred"] = model.fittedvalues
    for sub, subdf in data.groupby(subj_col):
        sea.regplot(data=subdf, x=dv_col, y="pred",
            color=color,
            ax=topax, **REGPLOT_ARGS)

    topax.set_xlim(.5, 9.5)
    topax.set_ylim(.5, 9.5)
    topax.axis("off")




plt.savefig(export_plot_filename)
plt.savefig(export_plot_filename.replace(".png", ".svg"))
plt.close()