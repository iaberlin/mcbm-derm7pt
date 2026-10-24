import argparse
import os

from dotenv import load_dotenv
from src.experiments import TrainExperiment, InterveneExperiment


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
    train = TrainExperiment(
        **vars(args), wandb_key=wandb_api_key)
    train.run()
    
    # only models taht have concepts do the intervention, i.e. vanilla models dont
    if getattr(train.model, "has_concepts", False):
        intervene = InterveneExperiment(
            **vars(args), wandb_key=wandb_api_key)
        intervene.run()