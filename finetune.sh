#!/usr/bin/env bash

# Prepare JAX CUDA runtime (local to this repo)
source "$(cd -- "$(dirname -- "$0")" && pwd)/env_jax_cuda.sh"

gpu="0,1,2,3,4,5,6,7"
pickle_path="./ckpt/BioSR_pretrain/pretrain_params.pkl"

# trainset="./data/SIM-simulation/*/*/train/*.tif"
# testset="./data/SIM-simulation/*/*/train/*.tif"
datatype="ring"
trainset="./data/single_data/${datatype}/train/*.tif"
testset="./data/single_data/${datatype}/train/*.tif"
save_dir="./ckpt/finetune/${datatype}_one_stack2"


CUDA_VISIBLE_DEVICES=${gpu} uv run python train.py \
        --trainset="${trainset}" \
        --testset="${testset}" \
        --batchsize=16 \
        --lr=1e-4 \
        --min_datasize=1540 \
        --sampling_rate=1 \
        --epoch=50 \
        --mask_ratio=0.25 \
        --add_noise=1 \
        --resume_pickle="${pickle_path}" \
        --save_dir="${save_dir}" \
        --patch_size 3 16 16 \
        --rescale 3 3 \
        --crop_size 80 80 \
        --psf_size 49 49 \
        --lrc=32 \
        --train_num_devices=8 \
        --use_gt \
        "$@" 2>&1 | tee training_log.txt
        # --mc_device_id=7 \
        # --mc_device='gpu' \
        # --mc_dropout_train_samples=8 \

fs1_path="./ckpt/finetune/beads/lr=0.0001--add_noise=1.0--lp_tv=0.001--mask_ratio=0.25--lrc=32--s1"

# CUDA_VISIBLE_DEVICES=${gpu} python train.py \
#         --trainset="${trainset}" \
#         --testset="${testset}" \
#         --batchsize=28 \
#         --lr=1e-4 \
#         --min_datasize=10000 \
#         --sampling_rate=1 \
#         --epoch=30 \
#         --mask_ratio=0.75 \
#         --add_noise=1 \
#         --resume_s1_path="${fs1_path}" \
#         --save_dir="${save_dir}" \
#         --patch_size 3 16 16 \
#         --rescale 3 3 \
#         --crop_size 80 80 \
#         --psf_size 49 49 \
#         --lrc=32 \
#         --not_resume_s1_opt \
#         --use_gt 2>&1 | tee training_log.txt


trainset="<path to dataset>/Open-3DSIM/*.tif"
testset="<path to dataset>/Open-3DSIM/*.tif"
save_dir="./ckpt/finetune/Open-3DSIM"


# CUDA_VISIBLE_DEVICES=${gpu} python train.py \
#         --trainset=${trainset} \
#         --testset=${testset} \
#         --batchsize=28 \
#         --lr=1e-4 \
#         --min_datasize=10000 \
#         --sampling_rate=1 \
#         --epoch=30 \
#         --mask_ratio=0.25 \
#         --add_noise=1 \
#         --resume_pickle="${pickle_path}" \
#         --save_dir=${save_dir} \
#         --patch_size 3 16 16 \
#         --rescale 3 3 \
#         --crop_size 80 80 \
#         --psf_size 49 49 \
#         --lrc=32 \
#         --adapt_pattern_dimension \
#         --target_pattern_frames=9 \
#         --random_pattern_sampling 2>&1 | tee training_log.txt

fs1_path="./ckpt/finetune/Open-3DSIM/lr=0.0001--add_noise=1.0--lp_tv=0.001--mask_ratio=0.25--lrc=32--s1"

# CUDA_VISIBLE_DEVICES=${gpu} python train.py \
#         --trainset=${trainset} \
#         --testset=${testset} \
#         --batchsize=28 \
#         --lr=1e-4 \
#         --min_datasize=10000 \
#         --sampling_rate=1 \
#         --epoch=30 \
#         --mask_ratio=0.75 \
#         --add_noise=1 \
#         --resume_s1_path=${fs1_path} \
#         --save_dir=${save_dir} \
#         --patch_size 3 16 16 \
#         --rescale 3 3 \
#         --crop_size 80 80 \
#         --psf_size 49 49 \
#         --lrc=32 \
#         --adapt_pattern_dimension \
#         --target_pattern_frames=9 \
#         --random_pattern_sampling \
#         --not_resume_s1_opt 2>&1 | tee training_log.txt
