#!/bin/bash
#SBATCH --job-name=hSS_annot
#SBATCH --account=jonesk-gpu-np      # edit for your allocation
#SBATCH --partition=jonesk-gpu-np
#SBATCH --gres=gpu
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=logs/annot_%j.out
#SBATCH --error=logs/annot_%j.err

# Pipeline: a1_qc (CPU) -> a2_tissue_rois (GPU, TITAN zero-shot v000) -> a2b_conch_patch_map (GPU, CONCH v1 patch map). After reviews: STEPS=a3_refine sbatch run_annot.slurm
set -euo pipefail
module use $HOME/MyModules
module load miniconda3/latest
source activate wsi_env
export HF_HOME=$HOME/.cache/huggingface
cd "$SLURM_SUBMIT_DIR"

STEPS=${STEPS:-"a1_qc a2_tissue_rois a2b_conch_patch_map"}
for s in $STEPS; do
  echo "=== $s $(date)"
  python -u $s.py
done
echo "=== ALL DONE $(date)"
