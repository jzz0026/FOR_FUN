#!/bin/bash
#SBATCH --job-name=omni_video_render
#SBATCH --partition=atlas
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/project/small_grains/Jinlie.Zhou/for_fun/Yong_lu/youtube_video/slurm_render_%j.log

set -euo pipefail
echo "=== Slurm Render Job $SLURM_JOB_ID on $(hostname) ($(nproc) CPUs) ==="
date
python3 /project/small_grains/Jinlie.Zhou/for_fun/Yong_lu/youtube_video/build_video_omni.py
date
echo "=== FINISHED ==="
