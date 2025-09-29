../../src/compute_embeddings_parallel \
    --input ./trees/chr21 \
    --output ./outputs/chr21_all_samples \
    --metadata ./pop_labels.csv \
    --min_time 100 --max_time 150000 --num_timepoints 200 --delta 200 --log_time --m 10