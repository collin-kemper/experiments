import random
import numpy as np

import torch

device = 'cuda'

block_size = 16
char_to_tok = {"0":0, "1":1}
meta_vocab_size = len(char_to_tok)
tok_to_char = {}
for (k, v) in char_to_tok.items():
    tok_to_char[v] = k


def tokenize(s):
    ret = np.zeros(block_size, dtype=np.int64)
    for (i, c) in enumerate(s):
        ret[i] = char_to_tok[c]
    return ret

def detokenize(Y):
    strs = []
    for batch_ind in range(Y.shape[0]):
        out = []
        for tok_ind in range(Y.shape[1]):
            out.append(tok_to_char[Y[batch_ind, tok_ind].item()])
        strs.append(''.join(out))
    return strs

def get_sample(a, b):
    c = a + b
    a_str = bin(a)[2:]
    b_str = bin(b)[2:]
    c_str = bin(c)[2:]
    x = "{:>08}{:>08}".format(a_str, b_str)
    y = "{:>016}".format(c_str)
    return torch.from_numpy(tokenize(x)), torch.from_numpy(tokenize(y))

def get_random_sample():
    num_bits_a = 8
    num_bits_b = num_bits_a
    a = random.randrange(0,1<<num_bits_a)
    b = random.randrange(0,1<<num_bits_b)
    return get_sample(a, b)

def get_single_batch(a, b):
    # data = train_data if split == 'train' else val_data
    x, y = get_sample(a, b)
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y


class WrongListLoader:
    def __init__(self, model, batch_size):
        self.model = model
        self.wrong_list = []
        self.using_wrong_list = False
        self.min_len_wrong_list = batch_size//12
        self.batch_size = batch_size

    def set_using_wrong_list(self, using_wrong_list):
        self.using_wrong_list = using_wrong_list

    def add_wrong(self, x, y):
        self.wrong_list.append((x, y))

    def get_sample(self):
        if self.using_wrong_list:
            if len(self.wrong_list) != 0:
                ret = self.wrong_list.pop()
                return ret
        return get_random_sample()

    def fill_wrong_list(self):
        self.model.eval()
        self.using_wrong_list = False
        iter_count = 0
        while len(self.wrong_list) < self.min_len_wrong_list:
            print(f"\tfill_wrong_list {iter_count}")
            if iter_count >= 100:
                break
            iter_count += 1

            X, Y = self.get_batch()
            logits, loss = self.model(X, Y)
            Xc, Yc = X.cpu(), Y.cpu()
            wrong_preds = (Y != logits.argmax(2)).sum(1).cpu()
            for (i, x) in enumerate(wrong_preds):
                if x.item() > 0:
                    self.wrong_list.append((Xc[i], Yc[i]))
        self.model.train()
        self.using_wrong_list = True

    def get_batch(self):
        if self.using_wrong_list and len(self.wrong_list) < self.min_len_wrong_list:
            self.fill_wrong_list()

        x_list = []
        y_list = []
        for _ in range(self.batch_size):
            x0,y0 = self.get_sample()
            x_list.append(x0)
            y_list.append(y0)

        x = torch.stack(x_list)
        y = torch.stack(y_list)

        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

        return x, y

class ExhaustiveLoader:
    def __init__(self):
        self.step_size = 1<<10
        self.a = 0
        self.b = 0

    def __iter__(self):
        return self

    def get_batch(self):
        # data = train_data if split == 'train' else val_data
        x_list = []
        y_list = []
        for _ in range(self.step_size):
            x0,y0 = get_sample(self.a, self.b)
            self.b += 1
            x_list.append(x0)
            y_list.append(y0)
        x = torch.stack([torch.from_numpy(x0) for x0 in x_list])
        y = torch.stack([torch.from_numpy(y0) for y0 in y_list])
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y

    def __next__(self):
        if self.b >= 1<<10:
            self.a += 1
            self.b = 0
        if self.a >= 1<<10:
            raise StopIteration
        return self.get_batch()

