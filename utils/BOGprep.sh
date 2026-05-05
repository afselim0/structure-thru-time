
### Open issues for discussion
### - Intuitions on number of time points 
###    - to my take - 200 time points is sufficent but how is that affecting the interpolation grid?
###    - Add parameter for n_timepoints embedding
### - Re-running chimp and Wohns to similar time-depths in the rates/embedding
### - Interpretations in the presence of admixture - source-pulling problem
###     - Admixture pull effecte change Nmax to 1e-6 or 1e-5
### - marida's code? 

### - Smoothing
### - badmixture are three cases -- bottlencck 3-way, and ghost
### - Marida's local admixture code on a branch
### - Marida's ancient samples might be ancients? 

##

# complex_4_pops
python plot_embeddings_dash.py \
    "../results/simulations/outputs/complex_4_pops/sim/data/embeddings.Nmin1000.Nmax1000000.L1[0.001].L2[0.01].minWS20.maxWS100.poly1.V1.FS1.0.maxT5000.nT500.parquet" \
    --labels ../results/simulations/outputs/complex_4_pops/sim/data/pop_labels.csv \
    --colors ../results/simulations/outputs/complex_4_pops/sim/data/pop_colors.csv \
    --rates ../results/simulations/outputs/complex_4_pops/sim/data/rates_true.parquet \
    --reconstructed_rates "../results/simulations/outputs/complex_4_pops/sim/data/reconstructed_rates.Nmin1000.Nmax1000000.L1[0.001].L2[0.01].minWS20.maxWS100.poly1.V1.FS1.0.maxT5000.nT500.parquet" \
    --N_max 1000000 \
    --time_scale linear \
    --min_time 20 \
    --max_time 3000

# stepping_stones_2d_Nm1
snakemake sim_STT --configfile ../data/demes/main_1_config/stepping_stones_2d_Nm1_config.yaml \
   --config output_path=../results/simulations/outputs/stepping_stones_2d_Nm1\
   --cores all --rerun-incomplete

python plot_embeddings_dash.py \
    "../results/simulations/outputs/stepping_stones_2d_Nm1/sim/data/embeddings.d2.Nmin100.Nmax10000000.L1[None].L2[None].minWS200.maxWS5000.V1.FS1.0.maxT100000.nT200.parquet" \
    --labels ../results/simulations/outputs/stepping_stones_2d_Nm1/sim/data/pop_labels.csv \
    --colors ../results/simulations/outputs/stepping_stones_2d_Nm1/sim/data/pop_colors.csv \
    --rates "../results/simulations/outputs/stepping_stones_2d_Nm1/sim/data/rates.minWS200.maxWS5000.V1.FS1.0.maxT100000.nT200.parquet" \
    --reconstructed_rates "../results/simulations/outputs/stepping_stones_2d_Nm1/sim/data/reconstructed_rates.d2.Nmin100.Nmax10000000.L1[None].L2[None].minWS200.maxWS5000.V1.FS1.0.maxT100000.nT200.parquet" \
    --N_max 10000000 \
    --time_scale linear \
    --min_time 50 \
    --max_time 100000

# Various demographies of interest
snakemake sim_STT --configfile ../data/demes/main_1_config/stepping_stones_2d_config.yaml \
   --config output_path=../results/simulations/outputs/stepping_stones_2d\
   --cores all --rerun-incomplete

snakemake sim_STT --configfile ../data/demes/main_1_config/admix_05_05_config.yaml \
   --config output_path=../results/simulations/outputs/admix_05_05\
   --cores all --rerun-incomplete

snakemake sim_STT --configfile ../data/demes/main_1_config/admix_01_09_config.yaml \
   --config output_path=../results/simulations/outputs/admix_01_09\
   --cores all --rerun-incomplete

snakemake sim_STT --configfile ../data/demes/main_1_config/ghost_admixture_config.yaml \
   --config output_path=../results/simulations/outputs/ghost_admixture\
   --cores all --rerun-incomplete

snakemake sim_STT --configfile ../data/demes/main_1_config/stepping_stones_2d_config.yaml \
   --config output_path=../results/simulations/outputs/stepping_stones_2d\
   --cores all --rerun-incomplete

snakemake sim_STT --configfile ../data/demes/main_config/OOA4_config.yaml \
   --config output_path=../results/simulations/outputs/OOA4\
   --cores all --rerun-incomplete

snakemake infer_STT --configfile ../data/demes/main_config/OOA4_config.yaml \
   --config output_path=../results/simulations/outputs/OOA4/tsinfer/ infer_method=tsinfer vcf_path=../results/simulations/outputs/OOA4/trees \
   --cores all --rerun-incomplete

snakemake sim_STT --configfile ../data/demes/main_1_config/split_sawtooth_config.yaml \
   --config output_path=../results/simulations/outputs/OOA4/tsinfer/ \
   --cores all --rerun-incomplete


   #snakemake infer_STT --configfile "$configfile" \
#  --config output_path="$output_path"/"$infer_method" infer_method="$infer_method" vcf_path="$output_path"/sim/trees \
#  --cores all


./run_sim_infer.sh -o ../results/simulations/outputs/OO4 -i tsinfer -p ../data/demes/main_config/OOA4_config.yaml


../results/simulations/outputs/split_sawtooth/plots/launch*.sh
../results/simulations/outputs/admix_01_09/plots/launch*.sh
../results/simulations/outputs/admix_01_09/plots/launch*.sh


python plot_embeddings_dash.py \
    "../results/simulations/outputs/admix_01_09/sim/data/embeddings.Nmin100.Nmax100000000.L1[None].L2[None].minWS100.maxWS5000.poly1.V1.FS1.0.maxT20000.nT5000.parquet" \
    --labels ../results/simulations/outputs/admix_01_09/sim/data/pop_labels.csv \
    --colors ../results/simulations/outputs/admix_01_09/sim/data/pop_colors.csv \
    --rates "../results/simulations/outputs/admix_01_09/sim/data/rates.minWS100.maxWS5000.poly1.V1.FS1.0.maxT20000.nT5000.parquet" \
    --reconstructed_rates "../results/simulations/outputs/admix_01_09/sim/data/reconstructed_rates.Nmin100.Nmax100000000.L1[None].L2[None].minWS100.maxWS5000.poly1.V1.FS1.0.maxT20000.nT5000.parquet" \
    --N_max 100000000 \
    --time_scale log \
    --min_time 50 \
    --max_time 20000


python plot_embeddings_dash.py \
    "../results/simulations/outputs/stepping_stones_2d/sim/data/embeddings.Nmin100.Nmax100000000.L1[None].L2[None].minWS100.maxWS5000.poly1.V1.FS1.0.maxT20000.nT5000.parquet" \
    --labels ../results/simulations/outputs/stepping_stones_2d/sim/data/pop_labels.csv \
    --colors ../results/simulations/outputs/stepping_stones_2d/sim/data/pop_colors.csv \
    --rates "../results/simulations/outputs/stepping_stones_2d/sim/data/rates.minWS100.maxWS5000.poly1.V1.FS1.0.maxT20000.nT5000.parquet" \
    --reconstructed_rates "../results/simulations/outputs/stepping_stones_2d/sim/data/reconstructed_rates.Nmin100.Nmax100000000.L1[None].L2[None].minWS100.maxWS5000.poly1.V1.FS1.0.maxT20000.nT5000.parquet" \
    --N_max 100000000 \
    --time_scale log \
    --min_time 50 \
    --max_time 20000



# DeManuel empirical
python evaluate_performance.py \
    --rates "../results/empirical/DeManuel/data/rates.minWS500.maxWS5000.V1.FS1.0.maxT100000.nT5000.parquet" \
    --reconstructed_rates "../results/empirical/DeManuel/data/performance.Nmin1.Nmax1000000000.L1[None].L2[None].minWS500.maxWS5000.V1.FS1.0.maxT100000.nT5000.parquet" \
    --labels ../data/metadata/DeManuel_pop_labels.csv \
    --colors ../data/metadata/DeManuel_pop_colors.csv \
    --N_max 1000000000 \
    --data_output_path ../results/empirical/DeManuel/data \
    --plot_output_path ../results/empirical/DeManuel/plots \
    --data_output_name "performance.Nmin1.Nmax1000000000.L1[None].L2[None].minWS500.maxWS5000.V1.FS1.0.maxT100000.nT5000.parquet" \
    --plot_output_name "performance.Nmin1.Nmax1000000000.L1[None].L2[None].minWS500.maxWS5000.V1.FS1.0.maxT100000.nT5000.png" \
    --min_time 250 \
    --time_scale log

python plot_embeddings_dash.py \
    "../results/empirical/DeManuel/data/embeddings.Nmin1.Nmax1000000000.L1[None].L2[None].minWS500.maxWS5000.V1.FS1.0.maxT100000.nT5000.parquet" \
    --labels ../data/metadata/DeManuel_pop_labels.csv \
    --colors ../data/metadata/DeManuel_pop_colors.csv \
    --rates "../results/empirical/DeManuel/data/rates.minWS500.maxWS5000.V1.FS1.0.maxT100000.nT5000.parquet" \
    --reconstructed_rates "../results/empirical/DeManuel/data/performance.Nmin1.Nmax1000000000.L1[None].L2[None].minWS500.maxWS5000.V1.FS1.0.maxT100000.nT5000.parquet" \
    --N_max 1000000000 \
    --time_scale log \
    --min_time 250 \
    --max_time 100000

### Human data
python evaluate_performance.py \
    --rates "../results/empirical/all.npp.2/data/rates.minWS100.maxWS1500.V1.FS1.0.maxT4000.nT4000.parquet" \
    --reconstructed_rates "../results/empirical/all.npp.2/data/reconstructed_rates.d2.Nmin1000.Nmax10000000.L1[None].L2[None].minWS100.maxWS1500.V1.FS1.0.maxT4000.nT4000.parquet" \
    --labels ../data/metadata/wohns_pop_labels.csv \
    --colors ../data/metadata/wohns_pop_colors.csv \
    --N_max 10000000 \
    --data_output_path ../results/empirical/all.npp.2/data \
    --plot_output_path ../results/empirical/all.npp.2/plots \
    --data_output_name "reconstructed_rates.d2.Nmin1000.Nmax10000000.L1[None].L2[None].minWS100.maxWS1500.V1.FS1.0.maxT4000.nT4000.parquet" \
    --plot_output_name "performance.d2.Nmin1000.Nmax10000000.L1[None].L2[None].minWS100.maxWS1500.V1.FS1.0.maxT4000.nT4000.png" \
    --min_time 100 \
    --time_scale linear

python plot_embeddings_dash.py \
"../results/empirical/all.npp.2/data/embeddings.d2.Nmin1000.Nmax10000000.L1[None].L2[None].minWS100.maxWS1500.V1.FS1.0.maxT4000.nT4000.parquet" \
--labels "../data/metadata/wohns_pop_labels.csv" \
--colors "../data/metadata/wohns_pop_colors.csv" \
--rates "../results/empirical/all.npp.2/data/rates.minWS100.maxWS1500.V1.FS1.0.maxT4000.nT4000.parquet" \
--reconstructed_rates "../results/empirical/all.npp.2/data/reconstructed_rates.d2.Nmin1000.Nmax10000000.L1[None].L2[None].minWS100.maxWS1500.V1.FS1.0.maxT4000.nT4000.parquet" \
--N_max 10000000 --max_time 10000

### Human data — nT200 maxT10000
snakemake STT --configfile ../data/empirical_runs/wohns_all_nT200_maxT10000.yaml \
    --config output_path=../results/empirical/all.npp.2 \
    --cores all --rerun-incomplete

python evaluate_performance.py \
    --rates "../results/empirical/all.npp.2/data/rates.minWS100.maxWS1500.V1.FS1.0.maxT10000.nT200.parquet" \
    --reconstructed_rates "../results/empirical/all.npp.2/data/reconstructed_rates.d2.Nmin1000.Nmax10000000.L1[None].L2[None].minWS100.maxWS1500.V1.FS1.0.maxT10000.nT200.parquet" \
    --labels ../data/metadata/wohns_pop_labels.csv \
    --colors ../data/metadata/wohns_pop_colors.csv \
    --N_max 10000000 \
    --data_output_path ../results/empirical/all.npp.2/data \
    --plot_output_path ../results/empirical/all.npp.2/plots \
    --data_output_name "reconstructed_rates.d2.Nmin1000.Nmax10000000.L1[None].L2[None].minWS100.maxWS1500.V1.FS1.0.maxT10000.nT200.parquet" \
    --plot_output_name "performance.d2.Nmin1000.Nmax10000000.L1[None].L2[None].minWS100.maxWS1500.V1.FS1.0.maxT10000.nT200.png" \
    --min_time 100 \
    --time_scale log

python plot_embeddings_dash.py \
    "../results/empirical/all.npp.2/data/embeddings.d2.Nmin1000.Nmax10000000.L1[None].L2[None].minWS100.maxWS1500.V1.FS1.0.maxT10000.nT200.parquet" \
    --labels "../data/metadata/wohns_pop_labels.csv" \
    --colors "../data/metadata/wohns_pop_colors.csv" \
    --rates "../results/empirical/all.npp.2/data/rates.minWS100.maxWS1500.V1.FS1.0.maxT10000.nT200.parquet" \
    --reconstructed_rates "../results/empirical/all.npp.2/data/reconstructed_rates.d2.Nmin1000.Nmax10000000.L1[None].L2[None].minWS100.maxWS1500.V1.FS1.0.maxT10000.nT200.parquet" \
    --N_max 10000000 \
    --time_scale log \
    --min_time 100 \
    --max_time 10000