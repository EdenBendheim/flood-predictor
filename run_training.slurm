#!/bin/bash

#=========================================================================================
# SLURM JOB SCRIPT FOR FLOOD PREDICTOR TRAINING
#=========================================================================================
#
#-----------------------------------------------------------------------------------------
# JOB CONFIGURATION
#-----------------------------------------------------------------------------------------
#
#SBATCH --job-name=flood-predictor-train   # A descriptive name for your job
#SBATCH --partition=npl-2024               # The partition to run on
#SBATCH --nodes=1                          # We need 6 GPUs, which fit on a single node
#SBATCH --ntasks=1                         # We are launching one main python script
#SBATCH --gres=gpu:6                       # Requesting 6 GPUs. This will also allocate 60 CPUs by default.
#SBATCH --mem=600G                         # Requesting 600 GB of system RAM
#SBATCH --time=5:55:00                    # Total run time limit (HH:MM:SS). Adjust as needed.
#
#-----------------------------------------------------------------------------------------
# EMAIL NOTIFICATIONS
#-----------------------------------------------------------------------------------------
#
#SBATCH --mail-type=BEGIN                  # Send email when the job begins
#SBATCH --mail-type=END                    # Send email when the job ends
#SBATCH --mail-type=FAIL                   # Send email if the job fails
#SBATCH --mail-user=bendhe@rpi.edu         # Your email address
#
#=========================================================================================
# JOB EXECUTION
#=========================================================================================

# Print job information
echo "==============================================================================="
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "==============================================================================="
echo ""

# Set up the environment (mirroring your interactive session)
echo "Loading modules..."
module load gcc
module load cuda/12.1
echo "Modules loaded."
echo ""

echo "Activating Conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cuda-flood
echo "Conda environment 'cuda-flood' activated."
echo ""

# Navigate to the project directory
# IMPORTANT: Use the full, absolute path to your project directory
PROJECT_DIR="/gpfs/u/scratch/MLFE/MLFEbndh/flood-predictor"
echo "Changing to project directory: $PROJECT_DIR"
cd $PROJECT_DIR
echo ""

# Run the training script
# The train.py script uses torch.multiprocessing.spawn, which will automatically
# detect and use all 6 of the allocated GPUs.
echo "Starting the Python training script..."
srun python train.py

echo "==============================================================================="
echo "Training script finished."
echo "Job complete."
echo "==============================================================================="