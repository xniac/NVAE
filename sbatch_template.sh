#!/bin/bash
#SBATCH --partition=atlas --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --job-name=xiaoyuan
#SBATCH --output=sample-%j.out

# only use the following if you want email notification
#SBATCH --mail-user=xniac@stanford.edu
#SBATCH --mail-type=ALL

# list out some useful information
echo "COMMAND: $@"
echo "PWD="$PWD
echo "CONDA_DEFAULT_ENV="$CONDA_DEFAULT_ENV
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "GPU CHECK:"
nvidia-smi
####echo "SLURMTMPDIR="$SLURMTMPDIR
####echo NPROCS=$NPROCS
echo "==============================================================================="
echo

$@

# done
echo
echo "==============================================================================="
echo "DONE"