#!/bin/bash

for cpoint in {10..100..10}
    do
        echo "Running for checkpoitn = $cpoint"
        CUDA_VISIBLE_DEVICES=0 python finetune.py --config configs/linear_probe/tuab_linear_probe_patch_100_alpha_powers.yaml\
            --checkpoint /home/gayal/ssl-project/gpatchTST/saved_models/pretrain/tuhab_pretrain_tuab_with_cls_token/TUH-101/2025-04-17_21-01-03/checkpoint_epoch_{$cpoint}.pth
    done
