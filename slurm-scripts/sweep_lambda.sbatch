#!/bin/bash

#SBATCH --job-name=sweep_lambda
#SBATCH --array=0-20
#SBATCH --output=slurm-%A_%a.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --chdir=/home/users/lbvh/josephson-network

echo "slurm script:"
echo "$(<$0)"

jjnetwork_dir=$HOME/josephson-network

echo "python script:"
echo "$(<./jjnetwork/scripts/compare_lambda.py)"

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate jjnetwork

outdir=$GROUP_SCRATCH/josephson-network/results/$SLURM_ARRAY_JOB_ID/
mkdir -p $outdir/logs/

python -m jjnetwork.scripts.compare_lambda \
    --directory=$outdir/$SLURM_ARRAY_TASK_ID \
    --Lambda-geomspace $Lambda_min $Lambda_max $SLURM_ARRAY_TASK_COUNT \
    --Lambda-index=$SLURM_ARRAY_TASK_ID \
    --geometry="square" \
    --lattice-constant=$lattice_constant \
    --fc-radius=$fc_radius \
    --pl-radius=$pl_radius \
    --fc-center 0 0 $fc_height \
    --pl-center 0 0 $fc_height \
    --fc-current="${fc_current} mA" \
    --patch-radius-factor=$rpatch \
    --starts=1

mv "slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" $outdir/logs/
