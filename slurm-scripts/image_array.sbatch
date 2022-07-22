

#!/bin/bash

#SBATCH --job-name=djn-image_array
#SBATCH --array=0-48
#SBATCH --output=slurm-%A_%a.out
#SBATCH --time=14:00:00
#SBATCH --ntasks=1
#SBATCH --mem=7G

echo "slurm script:"
echo "$(<$0)"

echo "python script:"
echo "$(<image_array.py)"

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate jjnetwork

nb_islands_dir=$HOME/nb-island-arrays/
jjnetwork_dir=$HOME/josephson-network/

outdir=$GROUP_SCRATCH/nb-island-arrays/results/$SLURM_ARRAY_JOB_ID/
mkdir -p $outdir/logs/

python image_array.py \
    --directory=$outdir/$SLURM_ARRAY_TASK_ID \
    --island-positions=$jjnetwork_dir/island-positions/"${position_file}.csv" \
    --fc-current="${fc_current} mA" \
    --length-units="um" \
    --fc-xs -42 42 $SLURM_ARRAY_TASK_COUNT \
    --fc-ys -42 42 $SLURM_ARRAY_TASK_COUNT \
    --fc-z=2 \
    --pl-z=-0.565 \
    --fc-radius=6.8 \
    --pl-radius=2.5 \
    --island-diameter="260 nm" \
    --junction-length-dependence="power_law" \
    --junction-I0="260 nA" \
    --junction-d0="240 nm" \
    --junction-cutoff-radius=0.9 \
    --patch-radius-factor=2.5 \
    --starts=10

mv "slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" $outdir/logs/
