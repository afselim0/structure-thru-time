#!/bin/bash
#SBATCH --job-name=chimp_genmap_idx
#SBATCH --account=pi-jnovembre
#SBATCH --partition=caslake
#SBATCH --qos=caslake
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:$PATH"
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"
cd "$SLURM_SUBMIT_DIR"

# genmap needs an uncompressed FASTA of just the 23 autosomes.
if [ ! -s resources/raw/panTro6_autosomes.fa ]; then
  python3 - <<'PY'
import gzip
keep={l.split()[0] for l in open('resources/panTro6/chrom.sizes')}
out=open('resources/raw/panTro6_autosomes.fa','w'); w=False; n=0
with gzip.open('resources/raw/panTro6.fa.gz','rt') as f:
    for line in f:
        if line.startswith('>'):
            name=line[1:].split()[0]; w = name in keep
            if w: out.write(f'>{name}\n'); n+=1
        elif w: out.write(line)
out.close(); print(f'wrote {n} records')
PY
fi

rm -rf resources/raw/genmap_idx
genmap index -F resources/raw/panTro6_autosomes.fa -I resources/raw/genmap_idx
mkdir -p resources/raw/genmap_out
genmap map -K 36 -E 0 -I resources/raw/genmap_idx -O resources/raw/genmap_out -t -bg
echo "genmap done"; ls -la resources/raw/genmap_out/ | head
