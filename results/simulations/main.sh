#!/bin/bash
    echo $f
    time snakemake --config demes_file_path=$f window_size=50 min_time=10 num_contigs=500 n_per_pop=20 contig_length=1000000 --cores all >> tmp.log
done


#for f in ../../data/demes/stdpopsim/*.yaml; do
#    echo $f
#    time snakemake --config demes_file_path=$f min_time=50 num_contigs=500 n_per_pop=20 contig_length=1000000 --cores all >> tmp.log
#done
