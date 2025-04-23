#!/bin/bash

#SBATCH --job-name=condaEnvSetup
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=1:00:00

# Load the conda module
export PATH=/opt/conda/bin:$PATH

# Create conda environment 
# and prevent the use of system site-package
# that will interfer with the conda env
export PYTHONNOUSERSITE=1

# Create the environment
conda env create -f environment.yml --force

# Activate the environment
source activate biodino

# Install the required libraries from a requirement file, using pypi
# pip install -r requirements.txt 
# You can also trigger "conda install ..." commands