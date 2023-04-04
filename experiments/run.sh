#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=1bf5
#SBATCH --gres=gpu:1

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate fasterrcnn-env                                 # activate target env

task="freeze4"
model="Freeze"
gpu=1
seed=100

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/fasterrcnn
python3 run.py \
    --random_seed ${seed} \
    --task_name ${task}"-"${model}"-gpu-"${gpu}"-seed-"${seed} \
    --model ${model} \
    --do_train \
    --num_gpu_devices ${gpu} \
    --freeze_depth 5