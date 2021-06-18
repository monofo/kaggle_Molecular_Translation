import os
import glob
import pandas as pd
import time
from datetime import datetime
import torch
import numpy as np
from utils import AverageMeter, load_tokenizer
from configure import *
from losses.loss_factory import np_loss_cross_entropy, seq_cross_entropy_loss, seq_anti_focal_cross_entropy_loss
import Levenshtein
from torch.cuda.amp import GradScaler, autocast
from optimizers.optimizer_factory import get_optimizer
from torch.optim.swa_utils import AveragedModel, SWALR

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np


import warnings

warnings.filterwarnings("ignore")

class PytorchTrainer:
    
    def __init__(self, model, device, config, tokenizer):
        self.config = config
        self.epoch = 0

        self.base_dir = f'./result/{config.dir}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        self.model = model
        self.device = device

        self.tokenizer = tokenizer

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        if 1:
            self.scaler = GradScaler()

        self.optimizer, self.scheduler = get_optimizer(self.model, self.config.optimizer_name, self.config.optimizer_params, 
                                                    self.config.scheduler_name, self.config.scheduler_params, self.config.n_epochs)

        if config.swa:
            self.swa_model = AveragedModel(model)
            self.swa_scheduler  = SWALR(self.optimizer, swa_lr=self.config.swa_lr, anneal_strategy="cos", anneal_epochs=10)

        self.log(f'Fitter prepared. Device is {self.device}')


    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            if not self.config.swa:
                summary_loss = self.train_one_epoch(train_loader)
            else:
                summary_loss = self.swa_train_one_epoch(train_loader)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')
            self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')

            # t = time.time()
            # summary_loss, summary_eva = self.validation(validation_loader)

            # self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss:.5f}, levenshtein: {summary_eva:.5f} time: {(time.time() - t):.5f}')
            # if summary_loss < self.best_summary_loss:
            #     self.best_summary_loss = summary_loss
            #     self.model.eval()
            #     self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
            #     for path in sorted(glob.glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
            #         os.remove(path)

            # if self.config.validation_scheduler:
            #     self.scheduler.step(metrics=summary_loss)

            self.epoch += 1
            

    def validation(self, val_loader):
        self.model.eval()

        valid_probability = []
        valid_truth = []
        valid_length = []
        valid_num = 0
        t = time.time()

        for step, batch in enumerate(val_loader):
            batch_size = len(batch['index'])
            image = batch["image"].to(self.device)
            token = batch["token"].to(self.device)
            length = batch["length"]

            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                logit = self.model(image, token, length)
                probability = torch.nn.functional.softmax(logit, -1)

            valid_num += batch_size
            valid_probability.append(probability.data.cpu().numpy())
            valid_truth.append(token.data.cpu().numpy())
            valid_length.extend(length)

        probability = np.concatenate(valid_probability)
        predict = probability.argmax(-1)
        truth = np.concatenate(valid_truth)
        length = valid_length

        p = probability[:, :-1].reshape(-1, vocab_size)
        t = truth[:, 1:].reshape(-1)

        non_pad = np.where(t!=STOI["<pad>"])[0]
        p = p[non_pad]
        t = t[non_pad]

        loss = np_loss_cross_entropy(p, t)
        lb_score = 0
        if 1:
            score = []
            for i, (p, t) in enumerate(zip(predict, truth)):
                t = truth[i][1:length[i]-1]     # in the buggy version, i have used 1 instead of i
                p = predict[i][1:length[i]-1]
                t = self.tokenizer.one_predict_to_inchi(t)
                p = self.tokenizer.one_predict_to_inchi(p)
                s = Levenshtein.distance(p, t)
                score.append(s)
            lb_score = np.mean(score)

        return loss, lb_score

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for step, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'Train Loss {summary_loss.avg}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            
            batch_size = len(batch['index'])
            image = batch["image"].to(self.device)
            token = batch["token"].to(self.device)
            length = batch["length"]

            with autocast():
                #assert(False)
                logit = self.model(image, token, length)
                loss0 = seq_cross_entropy_loss(logit, token, length)

            self.scaler.scale(loss0).backward()
            #scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            summary_loss.update(loss0)


        if self.config.step_scheduler:
            self.scheduler.step()

        return summary_loss.avg


    def swa_train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for step, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'Train Loss {summary_loss.avg}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            
            batch_size = len(batch['index'])
            image = batch["image"].to(self.device)
            token = batch["token"].to(self.device)
            length = batch["length"]

            with autocast():
                #assert(False)
                logit = self.model(image, token, length)
                loss0 = seq_anti_focal_cross_entropy_loss(logit, token, length)

            self.scaler.scale(loss0).backward()
            #scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            summary_loss.update(loss0)


        if self.config.step_scheduler and self.epoch >= int(self.config.n_epochs * 0.8):
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
        elif self.config.step_scheduler:
            self.scheduler.step()

        if self.epoch == self.config.n_epochs - 1:
            update_bn(train_loader, self.swa_model, self.device)
            self.swa_save(f'{self.base_dir}/swa_last-checkpoint.bin')

        return summary_loss.avg


    def predict(self, valid_loader, mode="submit", tta=True):
        text = []
        if mode != "submit":
            truths = []
            lengths = []
            image_ids = []


        t = time.time()
        valid_num = 0
        for step, batch in enumerate(valid_loader):
            if step % self.config.verbose_step == 0:
                print(
                    f'Train Step {step}/{len(valid_loader)}, ' + \
                    f'time: {(time.time() - t):.5f}', end='\r'
                )

            batch_size = len(batch['image'])
            image = batch['image'].cuda()
            
            if mode != "submit":
                token = batch["token"]
                length = batch["length"]

                truths.extend(token.detach().numpy())
                lengths.extend(length)
                image_ids.extend(batch["image_id"])

            self.model.eval()
            with torch.no_grad():
                with autocast():
                    if tta:
                        k,_ = self.do_tta(image)
                    else:
                        k = self.model.forward_argmax_decode(image)
                        k = k.detach().cpu().numpy()

                    # token = batch['token'].cuda()
                    # length = batch['length']
                    # logit = data_parallel(net,(image, token, length))
                    # k = logit.argmax(-1)

                    # k = k.detach().cpu().numpy()
                    k = self.tokenizer.predict_to_inchi(k)
                    text.extend(k)
            

            valid_num += batch_size
        # assert(valid_num == len(valid_loader.dataset))
        print('')

        if mode != "submit":
            score = []
            for i, (p, t) in enumerate(zip(text, truths)):
                t = truths[i][1:lengths[i]-1]     # in the buggy version, i have used 1 instead of i
                p = text[i]
                t = self.tokenizer.one_predict_to_inchi(t)
                # p = self.tokenizer.one_predict_to_inchi(p)
                s = Levenshtein.distance(p, t)
                score.append(s)
            lb_score = np.mean(score)

            df_submit = pd.DataFrame()
            df_submit.loc[:,'image_id'] = image_ids
            df_submit.loc[:,'InChI'] = text #
            df_submit.loc[:, "score"] = score

            return df_submit, lb_score

        return text


    def do_tta(self, x, **kwargs):
        # print(kwargs)
        preds1 = F.softmax(self.model.forward_argmax_decode(x, **kwargs), -1).cpu().numpy()
        # return preds1.argmax(-1), np.zeros((preds1.shape[0],))
        preds2 = F.softmax(self.model.forward_argmax_decode(x.flip(2), **kwargs), -1).cpu().numpy()
        # for me center crop works better.
        # x = TF.resize(TF.center_crop(x, (128, 224)), (320, 320))
        # preds3 = F.softmax(model.forward_argmax_decode(x, **kwargs), -1).cpu().numpy()
        preds3 = F.softmax(self.model.forward_argmax_decode(x.flip(3), **kwargs), -1).cpu().numpy()

        prob1, prob2 = np.zeros(preds1.shape[0],), np.zeros(
            preds2.shape[0],
        )
        prob3 = np.zeros(
            preds1.shape[0],
        )
        best_idx1, best_idx2 = (
            preds1.argmax(-1),
            preds2.argmax(-1),
        )
        best_idx3 = preds3.argmax(-1)
        pred_lengths1 = (best_idx1 == STOI['<eos>']).argmax(1)
        pred_lengths2 = (best_idx2 == STOI['<eos>']).argmax(1)
        pred_lengths3 = (best_idx3 == STOI['<eos>']).argmax(1)

        max_lengths = np.maximum(pred_lengths1, pred_lengths2, pred_lengths3)
        for b, l in zip(range(preds1.shape[0]), max_lengths):
            # prob1[b] = np.min(preds1[b, :pred_lengths1[b], best_idx1[b]])
            # prob2[b] = np.min(preds1[b, :pred_lengths2[b], best_idx2[b]])
            # prob3[b] = np.min(preds1[b, :pred_lengths3[b], best_idx3[b]])
            for seq in range(l):
                if seq < pred_lengths1[b]:
                    # prob1[b] = np.minimum(preds1[b, seq, best_idx1[b, seq]], prob1[b])
                    prob1[b] += preds1[b, seq, best_idx1[b, seq]]
                if seq < pred_lengths2[b]:
                    # prob2[b] = np.minimum(preds2[b, seq, best_idx2[b, seq]], prob2[b])
                    prob2[b] += preds2[b, seq, best_idx2[b, seq]]
                if seq < pred_lengths3[b]:
                    # prob3[b] = np.minimum(preds3[b, seq, best_idx3[b, seq]], prob3[b])
                    prob3[b] += preds3[b, seq, best_idx3[b, seq]]

        prob1 = prob1 / pred_lengths1
        prob2 = prob2 / pred_lengths2
        prob3 = prob3 / pred_lengths3
        preds = np.where((prob1 > prob2)[:, None], best_idx1, best_idx2)
        preds = np.where((prob3 > prob1)[:, None], best_idx3, preds)
        prob = np.maximum(prob1, prob2, prob3)
        return preds, prob
        

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)


    def swa_save(self, path):
        self.swa_model.eval()
        torch.save({
            'model_state_dict': self.swa_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)


    def load(self, path, jit=True, only_model=True):
        checkpoint = torch.load(path)
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        except:
            state_dict = checkpoint["model_state_dict"]
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            try:
                del state_dict["n_averaged"]
            except:
                pass
            self.model.load_state_dict(state_dict)
        
        if not only_model:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_summary_loss = checkpoint['best_summary_loss']
            self.epoch = checkpoint['epoch'] + 1
            if 1:
                self.swa_model = AveragedModel(self.model)
                self.swa_scheduler  = SWALR(self.optimizer, swa_lr=self.config.lr, anneal_strategy="cos", anneal_epochs=10)

        if jit:
            self.model = torch.jit.script(self.model)

        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')



@torch.no_grad()
def update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.
    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)
    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for step, batch in enumerate(loader):

        batch_size = len(batch['index'])
        image = batch["image"].to(device)
        token = batch["token"].to(device)
        length = batch["length"]

        model(image, token, length)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)