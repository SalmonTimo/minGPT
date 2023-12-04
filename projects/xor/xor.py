"""
Trains a GPT to xor n-bit numbers.
"""

import os
import sys
import json
from operator import xor

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

def fmt_bin(n, nbits):
    return bin(n)[2:].zfill(nbits)

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/xor'

    # data
    C.data = XORDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-nano'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

    return C

# -----------------------------------------------------------------------------

class XORDataset(Dataset):
    """
    Creates n-bit xor problems. For example, if n=2, then an example
    xor problem would be to compute 01 xor 11 = 10. This problem would be
    represented as the following string for the GPT:

    "011110"

    This is because:
    - we are discarding the xor and =, which are not necessary. We just encode the bits
      of the input numbers concatenated together.

    where you will notice that we are padding with zeros to make sure that we always
    produce strings of the exact same size: n + n + n. When n=2, this is 6.
    At test time, we will feed in an xor problem by giving the first 2n bits,
    and hoping that the GPT model completes the sequence with the next (n) bits
    correctly.
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.nbit = 5
        return C

    def __init__(self, config, split):
        self.config = config
        self.split = split # train/test

        # split up all xor problems into either training data or test data
        nbit = self.config.nbit
        assert nbit <= 50, "the lines below would be very memory inefficient, in future maybe refactor to support"
        num = (2**nbit)**2 # total number of possible xor problems with nbit strings
        rng = torch.Generator()
        rng.manual_seed(1337)
        perm = torch.randperm(num, generator=rng)
        num_test = min(int(num*0.2), 500) # 20% of the whole dataset, or only up to 500
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]

    def get_vocab_size(self):
        return 2 # bits 0, 1

    def get_block_size(self):
        # a, b, a xor b
        # but then also -1 because very last bit doesn't ever plug back
        # as there is no explicit <EOS> token to predict, it is implied
        return 3*self.config.nbit - 1

    def __len__(self):
        return self.ixes.nelement()

    def __getitem__(self, idx):
        nbit = self.config.nbit
        # given a problem index idx, first recover the associated a xor b
        idx = self.ixes[idx].item()
        nd = 2**nbit
        a = idx // nd
        b = idx %  nd
        # calculate the "label" of the addition problem a + b
        c = xor(a, b)
        # encode the bits of a, b, c into strings
        astr = fmt_bin(a, nbit)
        bstr = fmt_bin(b, nbit)
        cstr = fmt_bin(c, nbit)
        render = astr + bstr + cstr
        dix = [int(s) for s in render] # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long) # predict the next token in the sequence
        y[:(nbit*2)-1] = -1 # we will only train in the output locations. -1 will mask loss to zero
        # print('dix', dix)
        # print('x', x)
        # print('y', y)
        return x, y

# -----------------------------------------------------------------------------

def score_model_on_sequences(x, nbit, model):
    d1d2 = x[:, :nbit*2]
    # let the model sample the rest of the sequence
    d1d2d3 = model.generate(d1d2, nbit, do_sample=False) # using greedy argmax, not sampling
    # isolate the last bit of the sampled sequence
    d3 = d1d2d3[:, -(nbit):]

    d1 = d1d2[:, :nbit]
    d2 = d1d2[:, nbit:]
    d3_gt = torch.logical_xor(d1, d2).int() # manually calculate the ground truth
    return d1, d2, d3, d3_gt

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct train and test datasets
    train_dataset = XORDataset(config.data, split='train')
    test_dataset  = XORDataset(config.data, split='test')

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # helper function for the evaluation of a model
    def eval_split(trainer, split, max_batches=None):
        dataset = {'train':train_dataset, 'test':test_dataset}[split]
        nbit = config.data.nbit
        results = []
        mistakes_printed_already = 0
        correct_already_printed = 0
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
        for b, (x, y) in enumerate(loader):
            x = x.to(trainer.device)
            d1, d2, d3, d3_gt = score_model_on_sequences(x, nbit, model)
            # evaluate the correctness of the results in this batch
            correct = (d3 == d3_gt).all(dim=1) # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
            # print(correct)
            for i in range(x.size(0)):
                results.append(int(correct[i]))
                if not correct[i] and mistakes_printed_already < 5: # only print up to 5 mistakes to get a sense
                    mistakes_printed_already += 1
                    print("INCORRECT: GPT claims that %s xor %s = %s but gt is %s" % (d1[i], d2[i], d3[i], d3_gt[i]))
            if max_batches is not None and b+1 >= max_batches:
                break
        rt = torch.tensor(results, dtype=torch.float)
        print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
        return rt.sum()

    # iteration callback
    top_score = 0
    def batch_end_callback(trainer):
        global top_score

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            train_max_batches = {
                1: None,
                2: None,
                3: None,
                4: None,
                5: None,
                6: None,
                7: None,
                8: 500,
                9: 25,
                10: 5,
                50: 1000}[config.data.nbit] # if nbit=2 we can afford the whole train set, ow no
            model.eval()
            with torch.no_grad():
                train_score = eval_split(trainer, 'train', max_batches=train_max_batches)
                test_score  = eval_split(trainer, 'test',  max_batches=None)
            score = train_score + test_score
            # save the model if this is the best score we've seen so far
            if score > top_score:
                top_score = score
                print(f"saving model with new top score of {score}")
                ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()