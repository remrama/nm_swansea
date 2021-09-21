"""
separate dream and nondream content
create two new columns
one with bracketed one without

Also add a column that has the original/full
report, just by removing the brackets.

RAs went and surrounded nondream content with
{brackets}, and questionable material in {{double brackets}}.
Just remove it all here.
"""
import os
import re
import json

import pandas as pd


with open("./config.json","r") as config_file:
    P = json.load(config_file)
    DATA_DIRECTORY = P["data_directory"]

BASENAME = "NMStudy-DreamData-cleaned"

IMPORT_FNAME = os.path.join(DATA_DIRECTORY, f"{BASENAME}.xlsx")
EXPORT_FNAME = os.path.join(DATA_DIRECTORY, f"{BASENAME}_split.csv")

df = pd.read_excel(IMPORT_FNAME)


# fix stupid weird apostrophes
df["Dream report-Cleaned"] = df["Dream report-Cleaned"].str.replace("â€™", "'")


## make a copy of the original dream report without brackets

def remove_brackets(s):
    if pd.isnull(s):
        return pd.NA
    else:
        return s.replace("{","").replace("}","")

df["report"] = df["Dream report-Cleaned"].apply(remove_brackets)


## split the cleaned/bracketed report into dream and nondream content

# first, to make regex simpler, just replace
# the doublebrackets with single (there are only 6 instances anyways)
def double2single_brackets(s):
    if pd.isnull(s):
        return pd.NA
    else:
        return s.replace("{{","{").replace("}}","}")

df["Dream report-Cleaned"] = df["Dream report-Cleaned"].apply(double2single_brackets)

keep_regex = re.compile("\{(.*?)\}") # keep bracketed material
drop_regex = re.compile("[\{\[].*?[\}\]]") # drop bracketed material

def split_dream_report(s):
    if pd.isnull(s):
        return pd.NA, pd.NA
    else:
        # create a clean version of only dream content
        dream_str = re.sub(drop_regex, "", s)
        # create a clean version of only nondream content
        nondream_str = " ".join(re.findall(keep_regex, s))
        # cleanup multiple and trailing spaces upon return
        dream_str    = re.sub(" +", " ", dream_str).strip()
        nondream_str = re.sub(" +", " ", nondream_str).strip()
        return dream_str, nondream_str

df["dream"], df["nondream"] = zip(*df["Dream report-Cleaned"].apply(split_dream_report))


# get rid of any loose punctuation left over
def strip_xtra_punc(doc):
    if pd.isnull(doc):
        return pd.NA
    else:
        return " ".join([ tok for tok in doc.split() 
            if any([ char.isalpha() for char in tok ]) ])

df["report"] = df["report"].apply(strip_xtra_punc)
df["dream"] = df["dream"].apply(strip_xtra_punc)
df["nondream"] = df["nondream"].apply(strip_xtra_punc)


# export
df.to_csv(EXPORT_FNAME, index=False, na_rep="NA")
