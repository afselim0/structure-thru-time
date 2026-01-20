#!/bin/bash
for f in ../../data/demes/main/*.yaml; do
    echo $f
    time snakemake --config demes_file_path=$f min_time=50 --cores all 
done
