#!/bin/bash

#SBATCH --job-name=susc-thin-film
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
echo "$(<./jjnetwork/scripts/susc_thin_film.py)"

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate jjnetwork

outdir=$GROUP_SCRATCH/josephson-network/results/$SLURM_ARRAY_JOB_ID/
mkdir -p $outdir/logs/

python -m jjnetwork.scripts.susc_thin_film \
    --directory=$outdir/$SLURM_ARRAY_TASK_ID \
    --model="squid" \
    --squid-type="hypres.small" \
    --geometry="square" \
    --temp=$temp \
    --lambda0=$lambda0 \
    --thickness=$thickness \
    --xi0=$xi0 \
    --patch-radius-factor=4 \
    --squid-position 0 0 $squid_height \
    --squid-iterations=6 \
    --fc-current="${fc_current} mA" \
    --starts=$starts \
    --include-screening=$screening

mv "slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" $outdir/logs/
