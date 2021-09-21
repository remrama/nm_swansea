"""
Combine LIWC results with other data variables
and drop dream reports. Basically just get
back the original column names for the LIWC
results file and only retain the relevant stuff we look at.
"""
import os
import json

import pandas as pd


with open("./config.json","r") as config_file:
    P = json.load(config_file)
    DATA_DIRECTORY = P["data_directory"]


IMPORT_BASENAME = "data"

KEEP_COLUMNS = ["participant", "DRF", "BDRF", "NMRF",
    "report_lag", "sleep_efficiency", "word_count",
    "dream_negemo", "dream_posemo", "wake_negemo", "wake_posemo",
    "posemo", "anx", "anger", "sad",
    "work", "leisure", "home", "money", "relig", "death",
    "see", "hear", "feel", "body", "health", "sexual", "ingest",
    "family", "friend", "male", "female", "i", "we",
    "affiliation", "achieve", "power", "reward", "risk",
]

export_fname = os.path.join(DATA_DIRECTORY, f"{IMPORT_BASENAME}_anon.csv")

orig_fname = os.path.join(DATA_DIRECTORY, f"{IMPORT_BASENAME}.csv")
liwc_fname = os.path.join(DATA_DIRECTORY, f"LIWC2015 Results ({IMPORT_BASENAME}).csv")


df = pd.read_csv(liwc_fname)
orig_columns = pd.read_csv(orig_fname).columns
df.columns = orig_columns.append( df.columns[len(orig_columns):] )

df[KEEP_COLUMNS].round(2).to_csv(export_fname, index=False, na_rep="NA")
