# nm_swansea

Code for [Mallett et al., 2021, _Affective Science_](https://doi.org/10.1007/s42761-021-00080-8).

The same code, along with data, is on [the corresponding OSF repository](https://osf.io/2xy5p/).


## Linear description of scripts and analysis steps

Note that some important variables are defined in the `config.json` configuration file (e.g., data and results directory locations). All python packages and such are in the `environment.yaml` conda environment file.

```bash
####################################################################
############## ------ preprocessing (and LIWC) ------ ##############
####################################################################

# take raw data file and split into dream/nondream content
# while simultaneously converting from xlsx to csv
python preproc-split_report.py

# merge the data file with the demographics file
# while simultaneously doing some data cleanup and calculations
# (see comments in file for details about cleaning steps)
python preproc-clean_raw.py

#####
## run LIWC 2015 externally on the "dream" column of data.csv
## save with default filename (whatever LIWC wants to do)
#####

# make a clean data file that combines
# LIWC and relevant other variables but removes dreams
# (for online data repo, and will just use this for other analysis scripts)
python preproc-merge_liwc.py


####################################################
############## ------ analysis ------ ##############
####################################################

# save out some descriptives and a plot that shows how much data is left
python analysis-descriptives.py

# correlate self-report measures
python analysis-correlations.py

# run divergence analysis
python analysis-divergence.py

# run LIWC analysis
python analysis-liwc.py
```