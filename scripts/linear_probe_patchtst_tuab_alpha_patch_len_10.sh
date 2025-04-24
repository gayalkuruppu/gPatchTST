#!/bin/bash

for cpoint in {10..100..10}
    do
        echo "Running for checkpoitn = $cpoint"
        CUDA_VISIBLE_DEVICES=0 python finetune.py --config configs/linear_probe/tuab_normal_10_linear_probe_patch_10_alpha_powers.yaml\
            --checkpoint /home/gayal/ssl-project/gpatchTST/saved_models/pretrain/tuhab_pretrain_patch_len_10/TUH-111/2025-04-18_12-55-17/checkpoint_epoch_$cpoint.pth
    done
