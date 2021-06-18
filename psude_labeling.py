import argparse
import importlib
import os
import torch
import random
import numpy as np
import pandas as pd
from datasets.dataset import make_fold, BmsDataset, null_collate, FixNumSampler
from torch.utils.data import WeightedRandomSampler
from utils import load_tokenizer, read_pickle_from_file
from trainer import PytorchTrainer
from models.cnn_transfomer import Net as Net
from models.multiencoder_transfomer import Net as NetC


#************************************************
seed = 2021
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

seed_everything(seed)
device = torch.device('cuda')

#************************************************
import torch.cuda.amp as amp


class AmpNet(Net):
    @torch.cuda.amp.autocast()
    def forward(self, *args):
        return super(AmpNet, self).forward(*args)


def main(args):
    config = importlib.import_module(f"configs.{args.config}")

    train_df, valid_df = make_fold(config.mode)
    if 0:
        # Debug
        train_df = train_df[:200]
        valid_df = valid_df[:200]

        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)

    # merge best sub
    best_df = read_pickle_from_file("./data/df_test_small.csv.pickle")
    best_df = best_df[best_df["length"] != -1]

    best_df = best_df.rename({"InChI_1": "formula"}, axis=1)
    train_df = train_df.rename({"InChI_1": "formula"}, axis=1)
    valid_df = valid_df.rename({"InChI_1": "formula"}, axis=1)
    
    best_df.loc[:,'path']=f'test_320'
    best_df.loc[:, "orientation"] = 0
    best_df.loc[:, "fold"] = -1

    best_df = best_df[["image_id", "InChI", "formula", "text", "sequence", "length", "path", "orientation", "fold"]]
    train_df = pd.concat([train_df, best_df]).reset_index()
    print(train_df["length"].min(), train_df["length"].max())
    print(train_df.head())
    # train_df  = best_df
    
    tokenizer = load_tokenizer(is_load=config.tokenizer_path)

    train_dataset = BmsDataset(train_df,tokenizer, config.train_transforms)
    valid_dataset = BmsDataset(valid_df,tokenizer, config.valid_transforms)

    weights = train_df['length'].values
    train_sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset),
                                            replacement=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler = train_sampler,
        batch_size=config.batch_size,
        drop_last=True,
        num_workers=6,
        pin_memory=True,
        collate_fn=null_collate,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        #sampler=SequentialSampler(valid_dataset),
        sampler=FixNumSampler(valid_dataset, 5_000), #200_000
        batch_size=64,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=null_collate,
    )
    
    net = AmpNet(**config.model_params).cuda()
    
    runner = PytorchTrainer(model=net, device=device, config=config, tokenizer=tokenizer)

    if  config.is_load is not None:
        runner.load(config.is_load, jit=False, only_model=True)
    
    runner.fit(train_loader=train_loader, validation_loader=valid_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        required=True,
    )

    parser.add_argument(
        "--gpus",
        "-g",
        type=str,
        default="0"
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    main(args)