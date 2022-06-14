#!/bin/bash

#SBATCH --job-name=susc-two-loop
#SBATCH --array=0-9
#SBATCH --output=slurm-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --chdir=/home/users/lbvh/josephson-network


echo "slurm script:"
echo "$(<$0)"

jjnetwork_dir=$HOME/josephson-network

echo "python script:"
echo "$(<./jjnetwork/scripts/susceptibility.py)"

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate jjnetwork

outdir=$GROUP_SCRATCH/josephson-network/results/$SLURM_ARRAY_JOB_ID/
mkdir -p $outdir/logs/

python -m jjnetwork.scripts.susceptibility \
    --directory=$outdir/$SLURM_ARRAY_TASK_ID \
    --model="twoloop" \
    --geometry="square" \
    --lattice-constant=$lattice_constant \
    --junction-cutoff-radius=0.7 \
    --patch-radius-factor=3 \
    --fc-radius=6.8 \
    --pl-radius=2.5 \
    --fc-center 0 0 2 \
    --pl-center 0 0 1.5 \
    --fc-current="${fc_current} mA" \
    --junction-I0="260 nA" \
    --island-diameter="260 nm" \
    --junction-d0="${lattice_constant} nm" \
    --junction-length-dependence="power_law" \
    --starts=5

mv "slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" $outdir/logs/
