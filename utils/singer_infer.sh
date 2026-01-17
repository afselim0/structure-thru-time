#!/bin/bash
vcf_path=/Users/afselim/Library/CloudStorage/Box-Box/Novembre/Projects/structure_thru_time/results/simulations/outputs/split/trees/sim
contig_length=3e6
SINGER/releases/singer-0.1.8-beta-linux-x86_64/singer_master \
    -m 1e-8 \
    -vcf contig_0\
    -output contig_0\
    -start 0\
    -end 1e6\


SINGER_PATH=../../../../../../SINGER/SINGER/SINGER/singer_master

$SINGER_PATH -m 1e-8 -output contig_0 -start 0 -end 1e6 -vcf contig_0 -n 10 -thin 20