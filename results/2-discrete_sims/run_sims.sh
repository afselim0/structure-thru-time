#!/bin/bash
cd ../../src/

python ./compute_embeddings_sim --input ../results/2-discrete_sims/deme_files/zigzag.yaml --max_time 25000
python ./compute_embeddings_sim --input ../results/2-discrete_sims/deme_files/split.yaml --max_time 10000

python ./compute_embeddings_sim --input ../results/2-discrete_sims/deme_files/split_mig_Nm_01.yaml --max_time 10000
python ./compute_embeddings_sim --input ../results/2-discrete_sims/deme_files/split_mig_Nm_1.yaml --max_time 10000
python ./compute_embeddings_sim --input ../results/2-discrete_sims/deme_files/split_mig_Nm_10.yaml --max_time 10000

python ./compute_embeddings_sim --input ../results/2-discrete_sims/deme_files/ooa_4.yaml --max_time 25000

python ./compute_embeddings_sim --input ../results/2-discrete_sims/deme_files/admix_01_09.yaml --max_time 5000
python ./compute_embeddings_sim --input ../results/2-discrete_sims/deme_files/admix_03_07.yaml --max_time 5000
python ./compute_embeddings_sim --input ../results/2-discrete_sims/deme_files/admix_05_05.yaml --max_time 5000

python ./compute_embeddings_sim --input ../results/2-discrete_sims/deme_files/SS_N100_m_001.yaml --max_time 60000
python ./compute_embeddings_sim --input ../results/2-discrete_sims/deme_files/SS_N100_m_005.yaml --max_time 60000
python ./compute_embeddings_sim --input ../results/2-discrete_sims/deme_files/SS_N100_m_0005.yaml --max_time 60000