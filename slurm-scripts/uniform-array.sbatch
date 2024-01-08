#!/bin/bash

#SBATCH --job-name=uniform-array
#SBATCH --array=0-49
#SBATCH --output=slurm-%A_%a.out
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --mem=6G
#SBATCH --chdir=/home/users/lbvh/josephson-network


echo "slurm script:"
echo "$(<$0)"

jjnetwork_dir=$HOME/josephson-network

echo "python script:"
echo "$(<./jjnetwork/scripts/uniform_field.py)"

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate jjnetwork

outdir=$GROUP_SCRATCH/josephson-network/results/$SLURM_ARRAY_JOB_ID/
mkdir -p $outdir/logs/

python -m jjnetwork.scripts.uniform_field \
    --directory=$outdir/$SLURM_ARRAY_TASK_ID \
    --geometry="square" \
    --lattice-constant=0.5 \
    --Bz="4.353334417814589 mT" \
    --junction-cutoff-radius=0.7 \
    --array-size 30 30 \
    --junction-I0="200 nA" \
    --island-diameter="0 nm" \
    --junction-d0="500 nm" \
    --junction-length-dependence="power_law" \
    --starts=1

mv "slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" $outdir/logs/
