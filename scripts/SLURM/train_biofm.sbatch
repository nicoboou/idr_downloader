#!/bin/bash 

#SBATCH --job-name=biodino
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=03:00:00
#SBATCH --output=logslurms/slurm-%j.out
#SBATCH --error=logslurms/slurm-%j.err

# Load the conda module
export PATH=/opt/conda/bin:$PATH

# Load the conda environment
source activate biodino

current_dir=`pwd`

# Copying the source directory
echo "Copying the source directory..."

date
mkdir $TMPDIR/biodino

rsync -r --exclude 'biodino/models/*' /usr/users/bio_self_supervised/bourriezn/biodino/* $TMPDIR/biodino

echo "Directory copied !"

# Move to the temp directory
cd $TMPDIR/biodino/
export WANDB_KEY=$(cat config/wandb_key.txt)

wandb login $WANDB_KEY

python3 main.py --params_path='./config/params.json'

# Transfer output back on the disk
rsync -ravhzP $TMPDIR/biodino/models/* $current_dir/models/