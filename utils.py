import os
import numpy as np
import pickle
import Levenshtein

from rdkit import Chem
from rdkit import RDLogger
from configure import *
RDLogger.DisableLog('rdApp.*')

data_dir = './data'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def read_pickle_from_file(pickle_file):
    with open(pickle_file,'rb') as f:
        x = pickle.load(f)
    return x

def write_pickle_to_file(pickle_file, x):
    with open(pickle_file, 'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)

#https://www.kaggle.com/nofreewill/normalize-your-predictions

def normalize_inchi(inchi):
    try:
        mol = Chem.MolFromInchi(inchi)
        return inchi if (mol is None) else Chem.MolToInchi(mol)
    except: return inchi


# -----------------------------------------------------------------------

class YNakamaTokenizer(object):

    def __init__(self, is_load=None):
        self.stoi = {}
        self.itos = {}

        if is_load is not None:
            self.stoi = read_pickle_from_file(os.path.join(data_dir, is_load))
            self.itos = {k: v for v, k in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def build_vocab(self, text):
        vocab = set()
        for t in text:
            vocab.update(t.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {k: v for v, k in self.stoi.items()}

    def one_text_to_sequence(self, text):
        sequence = []
        sequence.append(self.stoi['<sos>'])
        for s in text.split(' '):
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def one_sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def one_predict_to_inchi(self, predict):
        inchi = 'InChI=1S/'
        for p in predict:
            if p == self.stoi['<eos>'] or p == self.stoi['<pad>']:
                break
            inchi += self.itos[p]
        return inchi

    # ---
    def text_to_sequence(self, text):
        sequence = [
            self.one_text_to_sequence(t)
            for t in text
        ]
        return sequence

    def sequence_to_text(self, sequence):
        text = [
            self.one_sequence_to_text(s)
            for s in sequence
        ]
        return text

    def predict_to_inchi(self, predict):
        inchi = [
            self.one_predict_to_inchi(p)
            for p in predict
        ]
        return inchi

def pad_sequence_to_max_length(sequence, max_length, padding_value):
    batch_size =len(sequence)
    pad_sequence = np.full((batch_size,max_length), padding_value, np.int32)
    for b, s in enumerate(sequence):
        L = len(s)
        pad_sequence[b, :L, ...] = s
    return pad_sequence


def load_tokenizer(is_load=None):
    tokenizer = YNakamaTokenizer(is_load=is_load)
    print('len(tokenizer) : vocab_size', len(tokenizer))
    for k,v in STOI.items():
        assert  tokenizer.stoi[k]==v
    return tokenizer

# -----------------------------------------------------------------------



def compute_lb_score(predict, truth):
    score = []
    for p, t in zip(predict, truth):
        s = Levenshtein.distance(p, t)
        score.append(s)
    score = np.array(score)
    return score

