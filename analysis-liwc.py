"""
linear mixed effects model with wake negative mood as outcome variable.
roll over a few different LIWC categories as predictors

Exports a couple tables that summarize model effects
with and without sleep efficiency.

Exports one plot showing LIWC effects on categories of interest.
"""
import os
import json

import numpy as np
import pandas as pd

import pingouin as pg

from scipy.stats import zscore
import statsmodels.formula.api as smf

import seaborn as sea
import matplotlib.pyplot as plt

plt.rcParams["savefig.dpi"] = 600
plt.rcParams["interactive"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"


with open("./config.json","r") as config_file:
    P = json.load(config_file)
    DATA_DIRECTORY = P["data_directory"]
    RESULTS_DIRECTORY = P["results_directory"]
    COLORS = P["colors"]


IMPORT_BASENAME = "data_anon.csv"

EXPORT_BASENAME = "liwc"

import_fname = os.path.join(DATA_DIRECTORY, IMPORT_BASENAME)
export_plot_fname = os.path.join(RESULTS_DIRECTORY, f"{EXPORT_BASENAME}.png")



df = pd.read_csv(import_fname)

OUTCOME = "wake_negemo"

CATEGORIES = {
    "affect" : ["posemo", "anx", "anger", "sad"],
    "personal_concerns" : ["work", "leisure", "home", "money", "relig", "death"],
    "sensory" : ["see", "hear", "feel", "body", "health", "sexual", "ingest"],
    "characters" : ["family", "friend", "male", "female", "i", "we"],
    "drives" : ["affiliation", "achieve", "power", "reward", "risk"],
}

LIWC_LABELS = {
    "posemo" : "pos emo",
    "anx" : "anxiety",
    "sad" : "sadness",
    "relig" : "religion",
    "ingest" : "ingestion",
}

PLOT_ORDER = ["affect", "personal_concerns", "characters",
    "drives", "sensory"]

all_categories = [ x for y in CATEGORIES.values() for x in y ]
assert len(set(all_categories)) == len(all_categories)

# zscore all variables, predictors and outcome, to get standardized betas
for p in all_categories:
    df[p] = zscore(df[p])
df[OUTCOME] = zscore(df[OUTCOME])


# have None as the last one to cycle over so that gets plotted
# (not super proud of this approach but that's okay)
for control_var in ["sleep_efficiency", None]:
    if control_var:
        export_table_fname = os.path.join(RESULTS_DIRECTORY, f"{EXPORT_BASENAME}+{control_var}.csv")
        df[control_var] = zscore(df[control_var], nan_policy="omit")
        analysis_df = df.dropna(subset=[control_var]).copy()
    else:
        export_table_fname = os.path.join(RESULTS_DIRECTORY, f"{EXPORT_BASENAME}.csv")
        analysis_df = df.copy()
    
    stats = {}

    for topcat, predictors in CATEGORIES.items():
        for pr in predictors:
            if control_var:
                formula = f"{OUTCOME} ~ {pr} + {control_var}"
            else:
                formula = f"{OUTCOME} ~ {pr}"
            model = smf.mixedlm(formula, analysis_df, groups="participant")
            result = model.fit()
            result_str = result.summary().as_text()

            z = result.tvalues[pr]
            p = result.pvalues[pr]
            b = result.params[pr]
            c = result.converged
            lo, hi = result.conf_int().loc[pr]
            result_dict = {
                "z" : z,
                "p" : p,
                "B" : b,
                "converged" : c,
                "superordinate_cat" : topcat,
                "5%" : lo,
                "95%" : hi,
                "model" : result_str
            }
            stats[pr] = result_dict

    res = pd.DataFrame.from_dict(stats,orient="index"
        ).rename_axis("LIWC_cat"
        ).sort_values("B",ascending=False)

    res["pass_alpha"], res["pcorr"] = pg.multicomp(res["p"].values,
        alpha=.05, method="fdr_bh")


    res.to_csv(export_table_fname, index=True, float_format="%.5f")



############# plot
############# plot
############# plot


def my_formatter(x, pos):
    """Format 1 as 1, 0 as 0, and all values whose absolute values is between
    0 and 1 without the leading "0." (e.g., 0.7 is formatted as .7 and -0.4 is
    formatted as -.4)."""
    val_str = '{:g}'.format(x)
    if np.abs(x) > 0 and np.abs(x) < 1:
        return val_str.replace("0", "", 1)
    else:
        return val_str

ERRORBAR_ARGS = {
    "capsize" : 2,
    "elinewidth" : 1,
    "fmt" : "o",
    "capthick" : 1,
    "markersize" : 4,
}

width_ratios = [ len(CATEGORIES[x]) for x in PLOT_ORDER ]
width_ratios = [ x/max(width_ratios) for x in width_ratios ]

fig, axes = plt.subplots(ncols=len(PLOT_ORDER),
    figsize=(6.7, 2.6),
    gridspec_kw=dict(width_ratios=width_ratios),
    constrained_layout=True)

for ax, topcat in zip(axes, PLOT_ORDER):

    subcats = CATEGORIES[topcat]
    tc_df = res.loc[subcats]
    tc_df = tc_df.sort_values("B", ascending=False)

    if ax.is_first_col():
        ylabel = r"$\beta$ predicting wake emotion" + "\n" + r"less negative$\leftarrow$$\rightarrow$more negative"
        ax.set_ylabel(ylabel, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set(major_locator=plt.MultipleLocator(.3),
                     minor_locator=plt.MultipleLocator(.1),
                     major_formatter=plt.FuncFormatter(my_formatter))
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set(major_locator=plt.NullLocator())

    xvals = np.arange(len(tc_df))
    xticklabels = tc_df.index.values
    xticklabels = [ LIWC_LABELS[x] if x in LIWC_LABELS else x for x in xticklabels ]
    yvals = tc_df["B"].values
    ci = tc_df[["5%", "95%"]].T.values
    ci = np.abs(ci-yvals)
    uncorrected_pvals = tc_df["p"].values

    pos_betas_mask = (yvals > 0) & (uncorrected_pvals < .05)
    neg_betas_mask = (yvals < 0) & (uncorrected_pvals < .05)
    no_effects_mask = (uncorrected_pvals >= .05)

    ax.errorbar(xvals[pos_betas_mask], yvals[pos_betas_mask], yerr=ci[:,pos_betas_mask], color=COLORS["negative"], **ERRORBAR_ARGS)
    ax.errorbar(xvals[neg_betas_mask], yvals[neg_betas_mask], yerr=ci[:,neg_betas_mask], color=COLORS["positive"], **ERRORBAR_ARGS)
    ax.errorbar(xvals[no_effects_mask], yvals[no_effects_mask], yerr=ci[:,no_effects_mask], color="gray", **ERRORBAR_ARGS)

    ax.axhline(0, color="silver", linestyle="dashed", linewidth=1, zorder=0)

    ax.set_xticks(xvals)
    ax.set_xticklabels(xticklabels, fontsize=8, rotation=33, ha="right")
    ax.tick_params(axis="x", which="major", pad=0)
    ax.tick_params(axis="y", which="major", labelsize=8)
    ax.set_ylim(-.3, .3)
    ax.set_xlim(xvals.min()-.5, xvals.max()+.5)

    ax.text(1, .95, topcat.replace("_","\n"), fontsize=8, color="black",
        ha="right", va="top", transform=ax.transAxes)

    for i, (_, row) in enumerate(tc_df.iterrows()):
        if row["B"] > 0:
            yloc = row["95%"] + .02
            va = "top"
            rot = 0
        else:
            yloc = row["5%"] - .02
            va = "bottom"
            rot = 180
        p = row["pcorr"]
        ha = "center"
        if p < .001:
            pchars = "***"
        elif p < .01:
            pchars = "**"
        elif p < .05:
            pchars = "*"
        else:
            pchars = ""
        ax.text(i, yloc, pchars, fontsize=12,
            va="center", ha=ha, rotation=rot)

XLABEL = "dream language"
axes[2].set_xlabel(XLABEL, fontsize=10)
axes[0].spines["left"].set_position(("outward", 15))


# export plot
plt.savefig(export_plot_fname)
plt.savefig(export_plot_fname.replace(".png", ".eps"))
plt.close()
