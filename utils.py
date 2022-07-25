import json
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from copy import deepcopy


def keep_k(x, k=100, absolute=True, dim=-1):
    shape = x.shape
    x_ = x
    if absolute:
        x_ = abs(x)
    values, indices = torch.topk(x_, k=k, dim=dim)
    res = torch.zeros_like(x)
    res.scatter_(dim, indices, x.gather(dim, indices))
    return res


def load_imdb():
    return load_dataset('imdb')['test']['text']
    
    
class TokenizerFromVocab:
    def __init__(self, vocab):
        self.vocab = vocab
    
    def convert_ids_to_tokens(self, arr):
        return [*map(vocab.__getitem__, arr.cpu().tolist())]
    
    def __len__(self):
        return len(self.vocab)

    
def get_multiberts_tokenizer():
    vocab = dict(enumerate(open('multiberts/vocab.txt', 'r').read().split('\n')[:-1]))
    return TokenizerFromVocab(vocab)


def convert_to_tokens(indices, tokenizer, strip=True, width=15):
    res = tokenizer.convert_ids_to_tokens(indices)
    if strip:
        res = list(map(lambda x: x[1:] if x[0] == 'Ġ' else "#" + x, res))
    if width:
        res = list(map(lambda x: x[:width] + (x[width:] and '...'), res))
    return res


def top_tokens(v, tokenizer, k=100, only_english=False, only_ascii=False, 
               exclude_brackets=False):
    v = deepcopy(v)
    ignored_indices = []
    if only_ascii:
        ignored_indices = [key for val, key in tokenizer.vocab.items() if not val.strip('Ġ').isascii()]
    if only_english: 
        ignored_indices =[key for val, key in tokenizer.vocab.items() 
                          if not (val.strip('Ġ').isascii() and val.strip('Ġ[]').isalnum())]
    if exclude_brackets:
        ignored_indices = set(ignored_indices).intersection(
            {key for val, key in tokenizer.vocab.items() if not (val.isascii() and val.isalnum())})
        ignored_indices = list(ignored_indices)
    v[ignored_indices] = -np.inf
    values, indices = torch.topk(v, k=k)
    res = convert_to_tokens(indices, tokenizer)
    return res


def top_matrix_tokens(mat, tokenizer, k=100, rel_thresh=None, thresh=None, 
                      sample_entries=10000, alphabetical=False, only_english=False,
                      exclude_brackets=False):
    mat = deepcopy(mat)
    ignored_indices = []
    if only_english:
        ignored_indices = [key for val, key in tokenizer.vocab.items() 
                           if not (val.isascii() and val.strip('[]').isalnum())]
    if exclude_brackets:
        ignored_indices = set(ignored_indices).intersection(
            {key for val, key in tokenizer.vocab.items() if not (val.isascii() and val.isalnum())})
        ignored_indices = list(ignored_indices)
    mat[ignored_indices, :] = -np.inf
    mat[:, ignored_indices] = -np.inf
    cond = torch.ones_like(mat).bool()
    if rel_thresh:
        cond &= (mat > torch.max(mat) * rel_thresh)
    if thresh:
        cond &= (mat > thresh)
    entries = torch.nonzero(cond)
    if sample_entries:
        entries = entries[np.random.randint(len(torch.nonzero(cond)), size=sample_entries)]
    res_indices = sorted(entries, key=lambda x: x[0] if alphabetical else -mat[x[0], x[1]])
    res = [*map(convert_to_tokens, res_indices)]
    return res
