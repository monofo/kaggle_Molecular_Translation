
import sys
import numpy as np
import cv2

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


# cnn_transfomer 

swa=True
tokenizer_path="small_tokenizer.stoi.pickle"
is_load="result/effv2_exp007/best-checkpoint-005epoch.bin"
hard_dir=None

dir="effv2_exp007_pl"
mode = "train_2_small2_320" # {mode}_{fold_num}_{tokenizer}_{img_size}

model_params = {
    "net_type": "efficientnetv2_rw_s",
    "image_dim": 1536,
    "text_dim": 1536,
    "decoder_dim": 1536,
    "num_layer": 3,
    "num_head": 12,
    "ff_dim": 1024,
    "num_pixel": 10*10,
}


num_workers=4
batch_size=86
n_epochs=15
img_size=(320, 320)
lr = 1e-4
swa_lr = 1e-6

optimizer_name = "radam"
optimizer_params = {
    "lr": lr,
    "weight_decay": 1e-6,
    # "momentum": 0.9,
    "opt_eps": 1e-8,
    "lookahead": False
}

scheduler_name = "CosineAnnealingWarmRestarts"
scheduler_params = {
    "warmup_factor": 0,
    "T_0": n_epochs-1,
    "T_multi": 1,
    "eta_min": 1e-6
}


####### data processings
train_transforms = A.Compose(
        [
            A.Resize(img_size[0], img_size[1], p=1.0),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=20,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=.5,
            ),
        ]
    )

valid_transforms = A.Compose([
            A.Resize(img_size[0], img_size[1], p=1.0),
        ], p=1.0)


# -------------------
verbose = True
verbose_step = 1
# -------------4------

# --------------------
step_scheduler = True  # do scheduler.step after optimizer.step
validation_scheduler = False  # do scheduler.step after validation stage loss