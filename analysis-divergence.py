"""
Exploring a more specific relationship between dream and wake moods.

Exports a stats table and a plot.

Exports a correlation thing too for a reviewer.
"""
import os
import json

import numpy as np
import pandas as pd

import pingouin as pg

import seaborn as sea
import matplotlib.pyplot as plt

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
EXPORT_BASENAME = "divergence"

import_filename = os.path.join(DATA_DIRECTORY, IMPORT_BASENAME)

export_stats_fname = os.path.join(RESULTS_DIRECTORY, f"{EXPORT_BASENAME}.csv")
export_plot_fname = os.path.join(RESULTS_DIRECTORY, f"{EXPORT_BASENAME}.png")
export_corr_fname = os.path.join(RESULTS_DIRECTORY, f"drf_corrs.csv")


# load data
df = pd.read_csv(import_filename)


#########################################################################
################  get divergence measure for each dream  ################
#########################################################################

def wake_dream_divergence(row, valence):
    """Quantify how much the wake emotion diverges from dream emotion.
    wake emotion minus dream emotion
    """
    assert valence in ["pos", "neg"]
    wake_emotion  = row[f"wake_{valence}emo"]
    dream_emotion = row[f"dream_{valence}emo"]
    divergence = wake_emotion - dream_emotion
    return divergence

df["neg_divergence"] = df.apply(wake_dream_divergence, axis=1, valence="neg")
df["pos_divergence"] = df.apply(wake_dream_divergence, axis=1, valence="pos")

# average per subject
pos_avg = df.groupby("participant").pos_divergence.mean()
neg_avg = df.groupby("participant").neg_divergence.mean()
avg_df = pd.merge(pos_avg, neg_avg, left_index=True, right_index=True)


##################################################
################  run statistics  ################
##################################################

stats_table = pd.concat(
    [pg.ttest(avg_df["neg_divergence"], 0, tail="two-sided"),
     pg.ttest(avg_df["pos_divergence"], 0, tail="two-sided"),
     pg.ttest(avg_df["neg_divergence"], avg_df["pos_divergence"], paired=True, tail="two-sided")],
    ignore_index=True)
stats_table.index = pd.Index(["neg","pos", "neg_vs_pos"], name="divergence")

# export stats table
stats_table.round(5).to_csv(export_stats_fname, index=True)



########################################
################  plot  ################
########################################

ERROR_KWS = {
    "zorder" : 3,
    "capsize" : 2,
    "elinewidth" : 1,
}

_, ax = plt.subplots(figsize=(2.5,5), constrained_layout=True)

y1, y1_err = avg_df["neg_divergence"].agg(["mean","sem"])
y2, y2_err = avg_df["pos_divergence"].agg(["mean","sem"])

ax.bar(0, y1, yerr=y1_err, color=COLORS["negative"],
    label="negative", width=.7, zorder=1, error_kw=ERROR_KWS)
ax.bar(1, y2, yerr=y2_err, color=COLORS["positive"],
    label="positive", width=.7, zorder=1, error_kw=ERROR_KWS)

sea.swarmplot(ax=ax, data=avg_df.melt(), x="variable", y="value",
    order=["neg_divergence", "pos_divergence"],
    palette=dict(neg_divergence=COLORS["negative"],pos_divergence=COLORS["positive"]),
    edgecolor="white", linewidth=.5, zorder=2,
    size=3)

# extract pvalues and mark
for x, v in enumerate(["neg", "pos"]):
    p = stats_table.loc[v,"p-val"]
    d = stats_table.loc[v, "cohen-d"]
    chars = "*" * sum([ p < cutoff for cutoff in [.05,.01,.001] ])
    d_str = f"{d:.2f}"
    d_str = rf"$d={d_str}$"
    if p < .001:
        p_str = r"$p<.001$"
    else:
        p_str = f"{p:.3f}".lstrip("0")
        p_str = rf"$p={p_str}$"
    total_str = p_str + "\n" + d_str
    ax.text(x, 1, chars, fontsize=14, ha="center", va="top", transform=ax.get_xaxis_transform())
    ax.text(x, .97, total_str, fontsize=8, ha="center", va="top", transform=ax.get_xaxis_transform())

# a e s t h e t i c s
ax.axhline(0, linestyle="dashed", linewidth=1, color="black", zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(-5, 5)
ax.set_xticks([0,1])
ax.set_xticklabels(["negative", "positive"])
ax.yaxis.set(major_locator=plt.MultipleLocator(1),
             minor_locator=plt.MultipleLocator(.2))
ax.set_xlabel("valence", fontsize=10)
ax.set_ylabel((r"$\Delta$ emotion intensity" + "\n"
    + r"lower in waking$\leftarrow$$\rightarrow$higher in waking  "),
    fontsize=10)


# export and close this plot
plt.savefig(export_plot_fname)
plt.savefig(export_plot_fname.replace(".png", ".svg"))
plt.close()



######################################################
################  correlation thingy  ################
######################################################

# correlate divergence with recall frequency for reviewer
avgs =df.groupby("participant")[
    ["DRF", "pos_divergence", "neg_divergence", ]].mean()

corrs = pg.pairwise_corr(avgs,
    columns=[["DRF",["neg_divergence","pos_divergence"]]],
    method="kendall")

corrs.to_csv(export_corr_fname, index=False)
