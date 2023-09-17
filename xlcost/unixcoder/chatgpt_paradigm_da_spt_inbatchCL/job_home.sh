#!/bin/bash
# DOCS (sbatch, srun)
# https://github.com/dasandata/Open_HPC/blob/master/Document/User%20Guide/5_use_resource/5.2_Allocate_Resource.md

# HOW TO USE SBATCH
# gpu 4개, 사용시간은 48시간으로 지정하여 jop.sh 작업제출
# 참고 : --time=일-시간:분:초
# (base 파티션 & base_qos 인 경우) sbatch --gres=gpu:4 --time=48:00:00 ./job.sh
# (big 파티션 & big_qos 인 경우)  sbatch -p big -q big_qos --gres=gpu:4 --time=48:00:00 ./job.sh
# (cpu 파티션 & cpu_qos 인 경우) sbatch -p cpu -q cpu_qos --time=48:00:00 ./job.sh
# (특정 디렉토리에 slurm 로그를 남기고 싶은 경우) sbatch --gres=gpu:1 --time=24:00:00 --output=/home/ysnamgoong42/ws/XLCoST/code/translation/code/slurm-out/slurm-%j-%x.out ./job.sh
# %j, %x 는 각각 jobID, jobName 을 의미함.

# HOW TO USE SRUN
# gpu 4개, 사용시간은 48시간으로 지정하여 bash 실행
# srun --gres=gpu:4 --time=48:00:00 --pty bash -i

# HOW TO CHECH THE STATE OF SBATCH
# squeuelong -u ysnamgoong42

# HOW TO CANCEL THE JOB
# : YOU CAN SEE THE JOB ID NUMBER BY SQUEUELONG
# scancel {jobID number}

#################################################################################################

echo ""
echo ""
echo ""

# echo "RUNNING SCRIPT: $SLURM_JOB_NAME"

# conda 환경 활성화.
source ~/.bashrc
conda activate xlcost

# # cuda 11.7 환경 구성.
# ml purge
# ml load cuda/11.7

# GPU 체크
nvidia-smi
nvcc -V

######################################## 작업 준비 끝 ############################################
# 활성화된 환경에서 코드 실행.


# sbatch (-p big -q big_qos) --gres=gpu:1 --time=24:00:00 --output=/home/ysnamgoong42/ws/ParaDA/unixcoder_test/slurm/slurm-%j-%x.out ./job.sh


for seed in 42 #43 44 45 46
do
    # Training
    python run_unixcoder_graphcodebert.py \
        --output_dir model_saved/${seed} \
        --model_name_or_path microsoft/unixcoder-base  \
        --do_train \
        --train_data_file dataset/train.jsonl \
        --eval_data_file dataset/valid.jsonl \
        --codebase_file dataset/valid.jsonl \
        --num_train_epochs 20 \
        --code_length 512 \
        --nl_length 128 \
        --train_batch_size 8 \
        --eval_batch_size 8 \
        --learning_rate 5e-5 \
        --seed ${seed}
        
    # Evaluating
    python run_unixcoder_graphcodebert.py \
        --output_dir model_saved/${seed} \
        --model_name_or_path microsoft/unixcoder-base  \
        --do_test \
        --test_data_file dataset/test.jsonl \
        --codebase_file dataset/test.jsonl \
        --num_train_epochs 20 \
        --code_length 512 \
        --nl_length 128 \
        --train_batch_size 8 \
        --eval_batch_size 8 \
        --learning_rate 5e-5 \
        --seed ${seed}

done


