# 2D runs, p=2
# snakemake STT_sim --configfile ../data/demes/SS_demes_config/SS_2d_N100_m05_p2.yaml --cores all --rerun-incomplete
# snakemake STT_sim --configfile ../data/demes/SS_demes_config/SS_2d_N100_m01_p2.yaml --cores all --rerun-incomplete
# snakemake STT_sim --configfile ../data/demes/SS_demes_config/SS_2d_N100_m001_p2.yaml --cores all --rerun-incomplete
# # 2D runs, p=1
# snakemake STT_sim --configfile ../data/demes/SS_demes_config/SS_2d_N100_m05_p1.yaml --cores all --rerun-incomplete 
# snakemake STT_sim --configfile ../data/demes/SS_demes_config/SS_2d_N100_m01_p1.yaml --cores all --rerun-incomplete
# snakemake STT_sim --configfile ../data/demes/SS_demes_config/SS_2d_N100_m001_p1.yaml --cores all --rerun-incomplete

# 1D runs, p=2
# snakemake STT_sim --configfile ../data/demes/SS_demes_config/SS_1d_N100_m05_p2.yaml --cores all --rerun-incomplete
# snakemake STT_sim --configfile ../data/demes/SS_demes_config/SS_1d_N100_m01_p2.yaml --cores all --rerun-incomplete
snakemake STT_sim --configfile ../data/demes/SS_demes_config/SS_1d_N100_m001_p2.yaml --cores all --rerun-incomplete
# 1D runs, p=1
# snakemake STT_sim --configfile ../data/demes/SS_demes_config/SS_1d_N100_m05_p1.yaml --cores all --rerun-incomplete
# snakemake STT_sim --configfile ../data/demes/SS_demes_config/SS_1d_N100_m01_p1.yaml --cores all --rerun-incomplete
snakemake STT_sim --configfile ../data/demes/SS_demes_config/SS_1d_N100_m001_p1.yaml --cores all --rerun-incomplete

