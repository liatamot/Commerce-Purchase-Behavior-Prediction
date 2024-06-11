import argparse
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.config import Config
from recbole.utils import init_seed
from recbole.utils.case_study import full_sort_topk
from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils import InputType, ModelType
from recbole.model.general_recommender import EASE

from recbole.quick_start import run_recbole

from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='/data/ephemeral/home/code/yaml/ease.yaml')
    parser.add_argument("--dataset", default="SASRec_dataset", type=str)

    # train args
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    set_seed(args.seed)

    config = Config(model='EASE',
                    config_file_list=[args.config_file],
                    dataset=args.dataset,
                    )
    
    print('Config loaded')
    
    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    dataset = create_dataset(config)
    
    run_recbole(
        model='EASE',
        dataset='SASRec_dataset',
        config_file_list=['/data/ephemeral/home/code/yaml/ease.yaml']
)

if __name__ == "__main__":
    main()