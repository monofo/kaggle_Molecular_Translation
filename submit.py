import cv2
import argparse
import importlib
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.optim.swa_utils import AveragedModel

from datasets.dataset import TestBmsDataset, FixNumSampler, make_fold, null_collate, BmsDataset
from models.cnn_transfomer import Net as Net
from trainer import PytorchTrainer
from utils import load_tokenizer, normalize_inchi, write_pickle_to_file, read_pickle_from_file
import Levenshtein

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


def fast_remote_unrotate_augment(r):
    image = r['image']
    image = cv2.resize(image, (320, 320))
    index = r['index']
    h, w = image.shape

    # if h > w:
    #     image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    l= r['d'].orientation
    if l == 1:
        image = np.rot90(image, -1)
    if l == 2:
        image = np.rot90(image, 1)
    if l == 3:
        image = np.rot90(image, 2)

    #image = cv2.resize(image, dsize=(image_size,image_size), interpolation=cv2.INTER_LINEAR)
    assert (320==320)


    image = image.astype(np.float16) / 255
    image = torch.from_numpy(image).unsqueeze(0).repeat(3,1,1)

    r={}
    r['image'] = image
    return r

#************************************************

def main(args):
    config = importlib.import_module(f"configs.{args.config}")

    mode = args.mode
    if mode == "submit":
        df_valid = make_fold('test')
        if 0:
            df_valid = df_valid[:170]
            
        df_valid = df_valid.sort_values('length').reset_index(drop=True)
        df_valid.rename({"InChI_1": "formula"}, axis=1)
        tokenizer = load_tokenizer(is_load=config.tokenizer_path)

        valid_dataset = TestBmsDataset(df_valid, tokenizer, fast_remote_unrotate_augment)
        
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            sampler = torch.utils.data.SequentialSampler(valid_dataset),
            batch_size=512,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
            # collate_fn=null_collate,
        )

    else:
        print("calc CV")
        train_df, df_valid = make_fold(config.mode)
        
        # num = 5_000 #200_000
        num = 200_000
        df_valid = df_valid[:num]

        tokenizer = load_tokenizer(is_load=config.tokenizer_path)
        valid_dataset = BmsDataset(df_valid, tokenizer, config.valid_transforms)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            # sampler=torch.utils.data.SequentialSampler(valid_dataset),
            sampler=FixNumSampler(valid_dataset, num), 
            batch_size=512,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=null_collate,
        )

    net = Net(**config.model_params).cuda()
    # net = torch.optim.swa_utils.AveragedModel(net)

    runner = PytorchTrainer(model=net, device=device, config=config, tokenizer=tokenizer)
    runner.load(f"result/{config.dir}/last-checkpoint.bin", jit=True)
    # runner.load(f"result/{config.dir}/best-checkpoint-002epoch.bin", jit=True)
    # runner.load(f"result/{config.dir}/swa_last-checkpoint.bin", jit=True)

    if mode == "submit":
        predict = runner.predict(valid_loader, mode="submit", tta=args.tta)
        # predict = read_pickle_from_file(f"result/{config.dir}/pre_text.pickle")

        if 0:
            predict = [normalize_inchi(t) for t in predict]  #

        df_submit = pd.DataFrame()
        df_submit.loc[:,'image_id'] = df_valid.image_id.values
        df_submit.loc[:,'InChI'] = predict #

        df_submit.to_csv(f"result/{config.dir}/submit.csv", index=False)
    else:
    
        df_cv, score = runner.predict(valid_loader, mode=args.mode, tta=args.tta)
        print("score = ", score)

        df_cv.to_csv(f"result/{config.dir}/submit_cv.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        required=True,
    )

    parser.add_argument(
        "--tta",
        "-t",
        action='store_true'
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="submit",
    )

    parser.add_argument(
        "--gpus",
        "-g",
        type=str,
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    main(args)
