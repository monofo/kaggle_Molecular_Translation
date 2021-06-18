import argparse
import importlib
import os
import torch
import random
import numpy as np
import pandas as pd
from datasets.dataset import make_fold, BmsDataset, null_collate, FixNumSampler
from torch.utils.data import WeightedRandomSampler
from utils import load_tokenizer
from trainer import PytorchTrainer
from models.cnn_transfomer import Net as Net


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

    if config.hard_dir is not None:
        hard_df = pd.read_csv("result/" + config.hard_dir)
        hard_df = hard_df[hard_df["score"] != 0].reset_index()
        train_df = train_df[train_df["image_id"].isin(hard_df.image_id.values)].reset_index()

    
    tokenizer = load_tokenizer(is_load=config.tokenizer_path)

    train_dataset = BmsDataset(train_df,tokenizer, config.train_transforms)
    valid_dataset = BmsDataset(valid_df,tokenizer, config.valid_transforms)

    train_df["weight_len"] = train_df["length"]/max(train_df["length"]) 
    # Weight for layers
    layers = train_df["InChI"].str.findall("/(.)").astype(str)
    layers_count = layers.map(layers.value_counts().to_dict())
    train_df["weight_layers"] = 1 - layers_count/len(layers_count)
    # Combined weights
    train_df["weights"] = train_df["weight_len"]*train_df["weight_layers"]
    train_sampler = WeightedRandomSampler(train_df["weights"], num_samples=len(train_dataset),
                                            replacement=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler = train_sampler,
        batch_size=config.batch_size,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=null_collate,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        #sampler=SequentialSampler(valid_dataset),
        sampler=FixNumSampler(valid_dataset, 5_000), #200_000
        batch_size=32,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=null_collate,
    )
    

    net = AmpNet(**config.model_params).cuda()
    runner = PytorchTrainer(model=net, device=device, config=config, tokenizer=tokenizer)

    if  0:
        runner.load("result/effv2_exp004/last-checkpoint.bin", jit=False, only_model=True)
    
    runner.fit(train_loader=train_loader, validation_loader=valid_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        required=True,
    )

    parser.add_argument(
        "--fold_num",
        "-fn",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gpus",
        "-g",
        type=str,
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    main(args)