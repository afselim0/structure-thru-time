#!/bin/bash
for f in ../../data/demes/main/*.yaml; do
    snakemake --config demes_file_path=$f --cores all
done
