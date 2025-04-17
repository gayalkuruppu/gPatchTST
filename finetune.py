from configs import Config
from models import PatchTST
from data import get_tuab_dataloaders
from utils.mask_utils import create_patches, create_mask, apply_mask


import os
import torch
import neptune
import argparse


def init_neptune(config):
    """
    Initialize a new Neptune run.
    """
    print("Creating a new Neptune run.")
    return neptune.init_run(
        project=config['neptune']['project'], 
        name=config['neptune']['experiment_name'],
        capture_stdout=False,  # Avoid duplicate logging of stdout
        capture_stderr=False   # Avoid duplicate logging of stderr
    )


def main():
    parser = argparse.ArgumentParser(description="Finetuning script for PatchTST.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--pretrained_model', type=str, required=True, help='Path to the pretrained model.')

    args = parser.parse_args()

    # Get available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config_file_path = args.config
    print(f"Using config file: {config_file_path}")
    # Load configuration
    config = Config(config_file=config_file_path).get()

if __name__=="__main__":
    main()