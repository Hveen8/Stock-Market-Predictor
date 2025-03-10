#!/bin/bash
#SBATCH --job-name=test     # create a short name for your job
#SBATCH --partition=dualcard # smallcard | midcard | dualcard | bigcard
#SBATCH --nodes=1                # node count - unles you are VERY good at what you're doing, you should keep this as-is
#SBATCH --ntasks-per-node=1               # total number of tasks across all nodes - you only have 1 node, so you only have 1 task. Leave this.
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks) - play with this number if you are using a lot of CPU, but most people are using these machines for GPU only
#SBATCH --gres=gpu:1 #most machines have a single GPU, so leave this as-is. If you are on a dual GPU partition, this can be changed to --gres=gpu:2 to use both
#SBATCH --mem-per-cpu=30G         # (12G) (30G) (72G) memory per cpu-core - unless you're doing something obscene the default here is fine. This is RAM, not VRAM, so it's like storage for your dataset
#SBATCH --time=72:00:00          # total run time limit - You can increase this however you wish, depending on your job's needs. However, it is a good idea to keep it to what you need, in case your job goes off
#the rails and you can't stop it, this will stop it automatically
#SBATCH --output=logs/slurm-%j.out


#If you are using your own custom venv, replace mine with yours. Otherwise, stick to this default. It has torch, transformers, accelerate and a bunch of others. I'm happy to add more common libraries
source /mnt/slurm_nfs/ece498_w25_20/LSTM_VENV_tmp__V3_10_12/LSTM_venv/bin/activate

##Checking if the GPU is actually being used
# srun nvidia-smi -l 1 &

module load cuda/12.3

#Trust. If you're using anything from huggingface, leave these lines it. These don't affect your job at all anyway, so really...just leave it in.
export TRANSFORMERS_CACHE=/local/cache
export HF_HOME=/local/cache
export SENTENCE_TRANSFORMERS_HOME=/local/cache

# ------------------
#      Run FIle
# ------------------
script="Model/run_tests.py"

echo "+======================================================================================+"
echo "              Runing Script: ${script}"
echo "+======================================================================================+"

start=$(date +%s)

#This is where you run your actual code. Here, we are just running python. Theoretically other stuff should work as well. If you are finding that the compute nodes don't have what you need, contact Mike.
python3 ${script}

end=$(date +%s)

runtime=$((end - start))
# Convert runtime (seconds) into days, hours, minutes, and seconds
days=$(( runtime / 86400 ))
rem=$(( runtime % 86400 ))
hours=$(( rem / 3600 ))
rem=$(( rem % 3600 ))
minutes=$(( rem / 60 ))
seconds=$(( rem % 60 ))


echo   "+===========================================+"
echo   "Runtime for: ${script}"
printf -- "-----------> %02dd:%02dh:%02dm:%02ds\n" "$days" "$hours" "$minutes" "$seconds"
echo   "+===========================================+"

#you activated a venv, so deactivate it when you're done
deactivate
