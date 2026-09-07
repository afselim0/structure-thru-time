snakemake STT_sim --configfile ../data/demes/main_1_config/two_admix.yaml --config LAI_method=True  --cores all
snakemake STT_sim --configfile ../data/demes/main_1_config/two_admix.yaml --config LAI_method=RFmix --cores all
snakemake STT_sim --configfile ../data/demes/main_1_config/two_admix.yaml --config LAI_method=GNN --cores all

snakemake STT_sim --configfile ../data/demes/main_1_config/three_way_admixture.yaml --config LAI_method=True --cores all
snakemake STT_sim --configfile ../data/demes/main_1_config/three_way_admixture.yaml --config LAI_method=RFmix --cores all
snakemake STT_sim --configfile ../data/demes/main_1_config/three_way_admixture.yaml --config LAI_method=GNN --cores all

# snakemake STT_sim --configfile ../data/demes/main_1_config/ghost_admixture.yaml --config LAI_method=True --cores all
snakemake STT_sim --configfile ../data/demes/main_1_config/ghost_admixture.yaml --config LAI_method=RFmix --cores all
snakemake STT_sim --configfile ../data/demes/main_1_config/ghost_admixture.yaml --config LAI_method=GNN --cores all

# snakemake STT_sim --configfile ../data/demes/main_1_config/recent_bottleneck.yaml --config LAI_method=True --cores all
snakemake STT_sim --configfile ../data/demes/main_1_config/recent_bottleneck.yaml --config LAI_method=RFmix --cores all
snakemake STT_sim --configfile ../data/demes/main_1_config/recent_bottleneck.yaml --config LAI_method=GNN --cores all