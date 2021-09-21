"""
Merge the dream data file 
with the participant info (questionnaire) file
so that everything is in one place.

Also do some cleaning up of some column names
to make analysis easier later.

Also also calculate some new columns (e.g., sleep efficiency).
"""
import os
import json

import numpy as np
import pandas as pd


with open("./config.json","r") as config_file:
    P = json.load(config_file)
    DATA_DIRECTORY = P["data_directory"]
    MIN_WORD_COUNT = P["minimum_word_count"]
    MIN_REPORT_COUNT = P["minimum_report_count"]


BASENAME_DATA         = "NMStudy-DreamData-cleaned_split.csv"
BASENAME_DEMOGRAPHICS = "NMStudy-QuestionnaireData+drmwrk.xlsx"

EXPORT_BASENAME = "data.csv"

INPUT_DATE_FORMAT = "%Y-%m-%d-%H-%M"

data_fname = os.path.join(DATA_DIRECTORY, BASENAME_DATA)
demo_fname = os.path.join(DATA_DIRECTORY, BASENAME_DEMOGRAPHICS)

export_fname = os.path.join(DATA_DIRECTORY, EXPORT_BASENAME)

data_df = pd.read_csv(data_fname)
demo_df = pd.read_excel(demo_fname)


# rename some of the demographics columns
DEMOGRAPHIC_RENAMINGS = {
    "Code"               : "participant",
    "NM group-for NIRS"  : "NM_nirs",
    "Dreamwork group (visit1-2)" : "dreamwork",
    "Visit 1 Date"       : "date_visit1",
    "Visit 2 Date"       : "date_visit2",
    "Visit 3 Date"       : "date_visit3",
    "VanAnxiety Total"   : "VanAnxT_visit1",
    "VanAnxietyTotal"    : "VanAnxT_visit2",
    "vanAnxiety Total"   : "VanAnxT_visit3",
    "VAN sub 1"          : "VanAnxS1_visit1",
    "Van sub 2"          : "VanAnxS2_visit1",
    "van sub 3"          : "VanAnxS3_visit1",
    "Van 1"              : "VanAnxS1_visit2",
    "Van 2"              : "VanAnxS2_visit2",
    "Van 3"              : "VanAnxS3_visit2",
    "Van 1.1"            : "VanAnxS1_visit3",
    "Van 2.1"            : "VanAnxS2_visit3",
    "Van 3.1"            : "VanAnxS3_visit3",
    "WELLBEING SUM"      : "wellbeing_visit1",
    "WELLBEING SUM.1"    : "wellbeing_visit2",
    "WELLBEING SUM.2"    : "wellbeing_visit3",
    "DLQ SUM"            : "lucidity_visit1",
    "DLQ SUM.1"          : "lucidity_visit2",
    "DLQ SUM.2"          : "lucidity_visit3",
    "ATD SUM"            : "atd_visit1",
    "ATD SUM.1"          : "atd_visit2",
    "ATD SUM.2"          : "atd_visit3",
}

DATA_RENAMINGS = {
    "SubID"                   : "participant",
    "DateTime-Psytoolkit"     : "report_date",
    "Dream report-Cleaned"    : "report_cleaned",
    "Negative Body Sensatino" : "dream_negbody",
    "Positive Body Sensation" : "dream_posbody",
    "Negative Emotion"        : "dream_negemo",
    "Positive Emotion"        : "dream_posemo",
    "Distress on waking"      : "wake_negemo",
    "Pos mood on waking"      : "wake_posemo",
}

# rename some of the data columns
demo_df = demo_df.rename(columns=DEMOGRAPHIC_RENAMINGS)
data_df = data_df.rename(columns=DATA_RENAMINGS)

df = pd.merge( data_df, demo_df, on="participant", how="inner" )



##############################################
###### drop reports with low word count ######
##############################################

# create a word count column by splitting each report
# into a list of words and then getting the length of that list
df["word_count"] = df["dream"].fillna("").str.split().apply(len)

pass_wc_indx = (df["word_count"] >= MIN_WORD_COUNT)

n_too_short = (pass_wc_indx^1).sum()
pct_too_short = 1 - pass_wc_indx.mean()
print(f"Removing {n_too_short} reports ({pct_too_short*100:.0f}% of total) for having less than {MIN_WORD_COUNT} words.")

# remove rows with a low word count
df = df[ pass_wc_indx ].reset_index(drop=True)



###################################################
###### label each row with its study "phase" ######
###################################################

# convert dates to python timedate objects to make them usable
df["report_date"] = pd.to_datetime(df["report_date"],format=INPUT_DATE_FORMAT)
df["date_visit1"] = pd.to_datetime(df["date_visit1"],format=INPUT_DATE_FORMAT)
df["date_visit2"] = pd.to_datetime(df["date_visit2"],format=INPUT_DATE_FORMAT)
df["date_visit3"] = pd.to_datetime(df["date_visit3"],format=INPUT_DATE_FORMAT)

# identify each row/report as coming from a certain study phase
def get_study_phase(row):
    """a function that marks each dream as
    coming from one of 3 time periods:
        before1   - before visit 1
        between12 - between visit 1 and visit 2
        between23 - between visit 2 and visit 3
        after1    - after visit 1 (without a later visit 2)
        after2    - after visit 2 (without a later visit 3)
        after3    - after visit 3
    """
    if row["report_date"] < row["date_visit1"]:
        return "before1"
    elif pd.isna(row["date_visit2"]):
        return "after1"
    elif row["report_date"] < row["date_visit2"]:
        return "between12"
    elif pd.isna(row["date_visit3"]):
        return "after2"
    elif row["report_date"] < row["date_visit3"]:
        return "between23"
    else:
        return "after3"

df["phase"] = df.apply(get_study_phase,axis=1)



########################################
###### calculate sleep efficiency ######
########################################

# fix WASO column by dropping this one cell with text
# df = df[ df["WASO"] != "20 but spread out through the night " ]
df["WASO"] = df["WASO"].replace({"20 but spread out through the night " : 20})
df["WASO"] = df["WASO"].astype(float)

# create sleep duration columns
## ok this is kinda overly-complicated,
## bc there is no day info, so we can't just subtract
## bedtime from risetime.
## by default it will just assign the datetime to today,
## so problematic if it becomes today at 23:00 for bedtime
## and then today at 07:00 for risetime
def time_in_bed(row):
    bedtime  = pd.to_datetime(row["Bedtime"])
    risetime = pd.to_datetime(row["Risetime"])
    if risetime < bedtime:
        risetime += pd.Timedelta("1D")
    timediff = risetime - bedtime
    # convert to minutes
    time_in_bed = timediff.seconds / 60
    return time_in_bed

def time_in_bed_awake(row):
    return row["WASO"] + row["SL"]

df["time_in_bed"] = df.apply(time_in_bed,axis=1)
df["time_in_bed_awake"] = df.apply(time_in_bed_awake,axis=1)

df["sleep_duration"] = df["time_in_bed"] - df["time_in_bed_awake"]

df["sleep_efficiency"] = 100 * (df["sleep_duration"]/df["time_in_bed"])



################################################
###### drop dreams without emotion recall ######
################################################

n0 = len(df)
df = df[ df["dream_negemo"] > 0 ]
df = df[ df["dream_posemo"] > 0 ]
df = df[ df["wake_negemo"] > 0 ]
df = df[ df["wake_posemo"] > 0 ]
n1 = len(df)

pct_left = n1/n0
print(f"{pct_left*100:.0f}% reports left after taking out those without emotion reports.")

###########################################
###### ??? other date/time stuff ??? ######
###########################################

#### sort user/datetime so that the next things work
#### the next functions assume this ordering
df.sort_values(["participant", "report_date"], ascending=True, inplace=True, ignore_index=True)

## add a column that denotes the dream sequence number
# (ie, if the subj has 5 dreams they will have 0-4 in chronological order)
user_counts = df["participant"].value_counts().to_dict()
report_counter = np.concatenate([ range(n) for user, n in sorted(user_counts.items()) ])
df["report_number"] = report_counter

# add some columns that show relative report times in days
df["days_from_last_entry"] = df.groupby("participant"
    )["report_date"].apply(
        lambda dt_series: dt_series.diff()
    ).dt.total_seconds().divide(60).divide(60).divide(24).fillna(0)
# df["days_from_first_entry"] = df.groupby("SubID"
#     )["report_date"].apply(
#         lambda dt_series: dt_series.diff().dt.total_seconds().cumsum()
#     ).divide(60).divide(60).divide(24).fillna(0)
df["days_from_first_entry"] = df.groupby("participant")["days_from_last_entry"].cumsum()

# label days for something?
df["weekday"] = df["report_date"].dt.weekday


#### estimate the time between waking up and report dream
def report_lag(row):
    report_ts = row["report_date"]
    rise_ts = pd.to_datetime(row["Risetime"])
    # correct rise timestamp bc it has no day info
    # (give it same day info as survey completion)
    rise_ts = rise_ts.replace(year=report_ts.year,
        month=report_ts.month, day=report_ts.day)
    # get difference in minutes
    report_lag = (report_ts - rise_ts).seconds / 60
    assert report_lag > 0
    return report_lag

df["report_lag"] = df.apply(report_lag, axis=1)



##################################################################
###### remove subjects without sufficient reports remaining ######
##################################################################

report_counts = df["participant"].value_counts()
passing_subjects = report_counts[report_counts >= MIN_REPORT_COUNT].index.tolist()
df = df[df["participant"].isin(passing_subjects)]
n_not_passing = len(report_counts) - len(passing_subjects)
print(f"{n_not_passing} subjects didn't have {MIN_REPORT_COUNT} reports left after other exclusions.")



#########################
###### save/export ######
#########################

df.to_csv(export_fname, index=False, na_rep="NA")
