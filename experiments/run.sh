#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --job-name=8scratch
#SBATCH --gres=gpu:1

# run the application
. /etc/profile.d/modules.sh                                   # Leave this line (enables the module command)
module purge                                                  # Removes all modules still loaded
source /jmain02/apps/python3/anaconda3/etc/profile.d/conda.sh # enable conda
conda activate fasterrcnn-env                                 # activate target env

task="freeze4"
model="Freeze"
gpu=1

cd /jmain02/home/J2AD015/axf03/yxz79-axf03/fasterrcnn
python3 run.py \
    --task_name ${task}"-"${model}"-gpu-"${gpu} \
    --model ${model} \
    --do_test \
    --ckpt_path "/jmain02/home/J2AD015/axf03/yxz79-axf03/fasterrcnn/checkpoints/3-8/freeze4-Freeze-gpu-8/freeze4-Freeze-gpu-8-date=3-8-epoch=03-val_loss=0.06.ckpt" \
    --log_every_n_steps 20 \
    --num_epochs 10 \
    --lr 1e-3 \
    --num_gpu_devices ${gpu} \
    --freeze_depth 4