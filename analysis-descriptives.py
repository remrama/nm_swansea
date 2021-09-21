"""
Export some descriptive tables of the LIWC categories of interest.

Also draw a plot that summarizes the dataset by showing
1. the number of subjects
2. the number of reports per subject
3. the average word count per report (per subject)
"""
import os
import json

import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams["savefig.dpi"] = 600
plt.rcParams["interactive"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"


with open("./config.json","r") as config_file:
    P = json.load(config_file)
    DATA_DIRECTORY = P["data_directory"]
    RESULTS_DIRECTORY = P["results_directory"]
    MIN_WORD_COUNT = P["minimum_word_count"]
    MIN_REPORT_COUNT = P["minimum_report_count"]


IMPORT_BASENAME = "data_anon.csv"
EXPORT_BASENAME = "descriptives"

import_filename = os.path.join(DATA_DIRECTORY, IMPORT_BASENAME)
export_plot_filename = os.path.join(RESULTS_DIRECTORY, f"{EXPORT_BASENAME}.png")
export_descr_filename1 = os.path.join(RESULTS_DIRECTORY, f"{EXPORT_BASENAME}-indiv.csv")
export_descr_filename2 = os.path.join(RESULTS_DIRECTORY, f"{EXPORT_BASENAME}-total.csv")


# load data
df = pd.read_csv(import_filename, index_col="participant")


###########################################################
###############  export summary stats  ####################
###############        (mean/sd)       ####################
###########################################################

# export 2 different tables
# 1. mean across all reports (unweighted)
# 2. mean across participants (after averaging within-subj first)
# (second one makes more sense)

indiv_descr = df.groupby("participant").mean().describe()
indiv_descr.insert(0, "n_dreams", df.index.value_counts().describe())

indiv_descr.T.round(2).to_csv(export_descr_filename1, index=True, index_label="variable")

# recall frequencies don't make sense here bc subjs have repeated values
total_descr = df.describe().drop(columns=["DRF", "BDRF", "NMRF"])

total_descr.T.round(2).to_csv(export_descr_filename2, index=True, index_label="variable")


###########################################################
########  plot number of reports and word count  ##########
###########################################################


BAR_ARGS = {
    "color" : "gainsboro",
    "width" : 1,
    "edgecolor" : "black",
    "linewidth" : .1,
}

SECOND_AXIS_COLOR = "tab:blue"
ERRORBAR_ARGS = {
    "color"           : SECOND_AXIS_COLOR,
    "markersize"      : 3,
    "elinewidth"      : .5,
    "ecolor"          : SECOND_AXIS_COLOR,
    "markeredgewidth" : 0,
    "markerfacecolor" : SECOND_AXIS_COLOR,
}


# how much data per subject and word counts
n_dreams = df.index.value_counts().rename("n_dreams")
wc = df.groupby("participant").agg({"word_count":["mean","sem"]})

# merge into one plot and order by dream count for visual aid
plot_df = pd.concat([n_dreams,wc], axis=1)
plot_df = plot_df.sort_values("n_dreams", ascending=False)

n_subjs = len(plot_df)

# pick x and y values
x = range(1, n_subjs+1)

y1    = plot_df["n_dreams"].values
y2    = plot_df[("word_count","mean")].values
y2err = plot_df[("word_count","sem")].values


# open figure with a twin axis
_, ax = plt.subplots(figsize=(3,3))
ax2 = ax.twinx() # instantiate a second axes that shares the same x-axis

# draw barplot of report counts on the left axis
ax.bar(x, y1, **BAR_ARGS)

# draw word count on the right axis
ax2.errorbar(x, y2, fmt=".", yerr=y2err, **ERRORBAR_ARGS)

# draw a dashed line to indicate the min word cutoff
ax2.axhline(MIN_WORD_COUNT, color=SECOND_AXIS_COLOR, linewidth=.5, linestyle="dashed")
ax.axhline(MIN_REPORT_COUNT, color="black", linewidth=.5, linestyle="dashed")

# aesthetics
ax.set_ylim(0,40)
ax2.set_ylim(0,200)
ax.set_ylabel("Number of dream reports")
ax.set_xlabel("Unique participant", labelpad=-4)
ax2.set_ylabel("Dream report length (word count)",
    rotation=270, labelpad=15, color=SECOND_AXIS_COLOR)
ax.set_xticks([1,n_subjs])
ax.yaxis.set(major_locator=plt.MultipleLocator(20),
             minor_locator=plt.MultipleLocator(10))
ax2.yaxis.set(major_locator=plt.MultipleLocator(100),
              minor_locator=plt.MultipleLocator(50))

ax2.spines["right"].set_edgecolor(SECOND_AXIS_COLOR)
ax2.tick_params(axis="y", which="both", colors=SECOND_AXIS_COLOR)


## some text with summary statistics
total = n_dreams.sum()
mean = n_dreams.mean()
std = n_dreams.std()
median = n_dreams.median()
mmin, mmax = n_dreams.min(), n_dreams.max()

stats_str = (f"{total} reports, {n_subjs} subjects\n\n"
    "reports per subject\n"
    f"mean (std) = {mean:.1f} ({std:.1f})\n"
    f"median = {median:.0f}\n"
    f"min/max = {mmin}/{mmax}"
)

ax.text(.98, .98, stats_str, transform=ax.transAxes,
    fontsize=8, va="top", ha="right")


plt.tight_layout()

plt.savefig(export_plot_filename)
plt.close()

