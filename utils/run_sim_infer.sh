#!/bin/bash
# Usage:
## Runs the full snakemake pipeline by simulating trees based on a demes file and inferring trees from the simulated vcfs
## Takes in path to a config file for the simulation parameters

output_path=""
infer_method=""
configfile=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output_path)
      output_path="$2"
      shift 2
      ;;
    -i|--infer_method)
      infer_method="$2"
      shift 2
      ;;
    -p|--parameters)
      configfile="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 -o <output_path> -i <infer_method> -p <configfile>"
      exit 1
      ;;
  esac
done

# Check for required arguments
if [[ -z "$output_path" ]] || [[ -z "$infer_method" ]] || [[ -z "$configfile" ]]; then
  echo "Usage: $0 -o <output_path> -i <infer_method> -p <configfile>"
  exit 1
fi


#snakemake STT_true --configfile "$configfile" \
#   --config output_path="$output_path"/sim\
#   --cores all --rerun-incomplete


# snakemake sim_STT --configfile "$configfile" \
#   --config output_path="$output_path"/sim\
#   --cores all --rerun-incomplete


snakemake infer_STT --configfile "$configfile" \
  --config output_path="$output_path"/"$infer_method" infer_method="$infer_method" vcf_path="$output_path"/sim/trees \
  --cores all


## Example
### ./run_sim_infer.sh -o ../results/simulations/outputs/split -i tsinfer -p ../data/demes/main_1_config/split_config.yaml
# ./run_sim_infer.sh -o ../results/simulations/outputs/split_sawtooth -i tsinfer -p ../data/demes/main_1_config/split_sawtooth_config.yaml
### ./run_sim_infer.sh -o ../results/simulations/outputs/complex_4_pops -i tsinfer -p ../data/demes/main_1_config/complex_4_pops_config.yaml
# ./run_sim_infer.sh -o ../results/simulations/outputs/stepping_stones_2d -i tsinfer -p ../data/demes/main_1_config/stepping_stones_2d_config.yaml