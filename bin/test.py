import argparse
import os

from dotenv import load_dotenv
from src.experiments import TestExperiment


def parse_args():

    parser = argparse.ArgumentParser(description='Train model')

    parser.add_argument(
        'config_file', 
        nargs='?', 
        type=str,
        help='configuration file'
    )

    parser.add_argument(
        '-s', '--seed', 
        type=int, 
        default=42, 
        help='seed for initialization'
    )

    parser.add_argument(
    '--split_seed',
    type=int,
    default=42,
    help='fixed seed for data filtering and train/test split',
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    experiment = TestExperiment(
        **vars(args), wandb_key=wandb_api_key)
    experiment.run()