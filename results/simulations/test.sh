#!/bin/bash

# snakemake sim --config demes_file_path=../../data/demes/main/split.yaml N0=1 min_time=200 --cores all
# snakemake --config demes_file_path=../../data/demes/stdpopsim/OutOfAfrica_4J17.yaml n_timepoints=100 window_size=500 --cores all
# snakemake --config demes_file_path=../../../data/demes/main/SS_N100_m_001.yaml N0=1 --cores all
snakemake sim --config \
 demes_file_path=../../data/demes/main/zigzag.yaml \
 max_time=40000 \
 n_timepoints=4000 \
 minimum_window_size=51 \
 maximum_window_size=501 \
 n_per_pop=20 \
 --cores all
