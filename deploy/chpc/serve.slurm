#!/bin/bash
#SBATCH --job-name=hSS_review
#SBATCH --account=jonesk-gpu-np      # edit for your allocation
#SBATCH --partition=jonesk-gpu-np
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/review_%j.out
#SBATCH --error=logs/review_%j.err

# Review UI server. Reach it with a tunnel through the existing login connection:
#   ssh -O forward -L 8765:<node>:8765 chpc-login     then open http://localhost:8765/?token=$(cat ~/.hss_annot_token)
set -euo pipefail
module use $HOME/MyModules
module load miniconda3/latest
source activate wsi_env
cd "$SLURM_SUBMIT_DIR/app"
PORT=${PORT:-8765}
echo "node $(hostname) port $PORT"
uvicorn server:app --host 0.0.0.0 --port $PORT --workers 1
