#!/bin/bash
SINGER_PATH=../../SINGER/SINGER/SINGER

while getopts "p:m:n:c:" opt; do
    case $opt in
        p) parent_dir="$OPTARG" ;;
        m) num_posterior_samples="$OPTARG" ;;
        n) num_contigs="$OPTARG" ;;
        c) contig_length="$OPTARG" ;;
        *) echo "Usage: $0 -p parent directory path -m num posterior samples -n num_contigs -c contig_length"  >&4; exit 1 ;;
    esac
done

vcf_path=$parent_dir/sim
singer_trees_path=$parent_dir/singer

run_singer() {
    contig_id=$1
    $SINGER_PATH/singer_master -m 1e-8 -output $singer_trees_path/contig_$contig_id -start 0 -end $contig_length -vcf $vcf_path/contig_$contig_id -n $num_posterior_samples -thin 20
    $SINGER_PATH/convert_to_tskit -input $singer_trees_path/contig_$contig_id -output $singer_trees_path/contig_$contig_id -start 0 -end $num_posterior_samples
}

export -f run_singer
export SINGER_PATH
export vcf_path
export singer_trees_path
export num_posterior_samples
export contig_length

seq 0 $((num_contigs - 1)) | xargs -n 1 -P $(nproc) -I {} bash -c 'run_singer {}'