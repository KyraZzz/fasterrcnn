#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=dev
#SBATCH --gres=gpu:1

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate fasterrcnn-env                                 # activate target env

model="FasterRCNN"

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/fasterrcnn
python3 run.py \
    --task_name "dev-"${model} \
    --model ${model} \
    --log_every_n_steps 20 \
    --max_epoch 100 \
    --lr 1e-3 \
    --num_gpu_devices 1