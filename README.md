# gPatchTST

python pretrain.py --config configs/test_pretrain_repo_tuh.yaml --checkpoint saved_models/pretrain/tuh_pretrain/TUH-85/2025-04-14_21-33-45/checkpoint_epoch_100.pth

python preprocessing/scalogram_generation.py --input_dir /mnt/ssd_4tb_0/data/tuhab_preprocessed --num_workers 1 --image_height 70 --time_window 5