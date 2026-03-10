#!/bin/bash

PATH_TO_CONFIG_FILES=../data/demes/main_config
PARENT_OUTPUT_DIR=../results/simulations/outputs
INFER_METHOD=tsinfer

for configfile in "$PATH_TO_CONFIG_FILES"/*; do
  # Try to extract "demography:" from the YAML config
  demography=$(grep -E '^(demography|dempgraphy):' "$configfile" | head -n1 | awk '{print $2}')
  
  # Optionally, handle missing demography
  if [[ -z "$demography" ]](); then
    echo "Could not find demography/dempgraphy label in $configfile"
    continue
  fi

  echo "Running run_sim_infer.sh on $configfile (demography: $demography)"
  ./run_sim_infer.sh -o "$PARENT_OUTPUT_DIR/$demography" -i "$INFER_METHOD" -p "$configfile"
done