#!/bin/bash

snakemake --config demes_file_path=../../data/demes/main/split.yaml N0=1 min_time=200 --cores all
# snakemake --config demes_file_path=../../../data/demes/stdpopsim/OutOfAfrica_4J17.yaml --cores all
# snakemake --config demes_file_path=../../../data/demes/main/SS_N100_m_001.yaml N0=1 --cores all
# snakemake --config demes_file_path=../../../data/demes/main/zigzag.yaml N0=1000 fraction=1 --cores all
