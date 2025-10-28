#!/bin/bash
demes_dir=../../data/main_deme_files

# # Simulate Trees
python ./sim_chrom.py --input $demes_dir/zigzag.yaml --max_time 40000
# python ./sim_chrom.py --input $demes_dir/split.yaml --max_time 10000
# python ./sim_chrom.py --input $demes_dir/split_mig_Nm_1.yaml --max_time 10000
# python ./sim_chrom.py --input $demes_dir/mig_change.yaml --max_time 10000
# python ./sim_chrom.py --input $demes_dir/SS_N100_m_001.yaml --max_time 60000
python ./sim_chrom.py --input ../../local_data/deme_files/admix_05_05.yaml --max_time 10000

# Run method on simulated trees
python ../../src/compute_embeddings --input trees/zigzag --max_time 40000 --metadata trees/zigzag/metadata.csv --output outputs/main/zigzag --delta 100 --num_timepoints 500
python ../../src/compute_embeddings --input trees/split --max_time 20000 --metadata trees/split/metadata.csv --output outputs/main/split --no_log_time
python ../../src/compute_embeddings --input trees/split_mig_Nm_1 --max_time 10000 --metadata trees/split_mig_Nm_1/metadata.csv --output outputs/main/split_mig_Nm_1 --no_log_time --num_timepoints 500
python ../../src/compute_embeddings --input trees/mig_change --max_time 10000 --metadata trees/mig_change/metadata.csv --output outputs/main/mig_change --num_timepoints 500
python ../../src/compute_embeddings --input trees/SS_N100_m_001 --max_time 60000 --metadata trees/SS_N100_m_001/metadata.csv --output outputs/main/SS_N100_m_001
# python ../../src/compute_embeddings --input trees/admix_05_05 --max_time 10000 --metadata trees/admix_05_05/metadata.csv --output outputs/main/admix_05_05
