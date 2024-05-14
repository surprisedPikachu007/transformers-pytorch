import torch
import os
import codecs
import youtokentome
import math
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data(data_folder, min_length=3, max_length=100,
                 max_length_ratio=1.5, retain_case=True):
    # BPE
    print("\nLearning BPE...")
    youtokentome.BPE.train(data=os.path.join(data_folder, "tam-eng.tsv"), vocab_size=37000,
                            model=os.path.join(data_folder, "bpe.model"))
    
    # Load BPE model
    print("\nLoading BPE model...")
    bpe_model = youtokentome.BPE(model=os.path.join(data_folder, "bpe.model"))

    # Read English, Tamil
    print("\nRe-reading single files...")
    with codecs.open(os.path.join(data_folder, "tam-eng.tsv"), "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Filter
    print("\nFiltering...")
    pairs = list()
    for line in tqdm(lines, total=len(lines)):
        en, ta = line.strip().split('\t')
        en_tok = bpe_model.encode(en, output_type=youtokentome.OutputType.ID)
        ta_tok = bpe_model.encode(ta, output_type=youtokentome.OutputType.ID)
        len_en_tok = len(en_tok)
        len_ta_tok = len(ta_tok)

        if min_length < len_en_tok < max_length and \
                min_length < len_ta_tok < max_length and \
                1. / max_length_ratio <= len_ta_tok / len_en_tok <= max_length_ratio:
            pairs.append((en, ta))
        else:
            continue

    # Rewrite into train, test, val files
    english, tamil = zip(*pairs)
    print("\nRe-writing filtered sentences to single files...")
    with codecs.open(os.path.join(data_folder, "train.en"), "w", encoding="utf-8") as f:
        f.write("\n".join(english[:int(0.8*len(english))]))
    with codecs.open(os.path.join(data_folder, "train.ta"), "w", encoding="utf-8") as f:
        f.write("\n".join(tamil[:int(0.8*len(tamil))]))

    with codecs.open(os.path.join(data_folder, "val.en"), "w", encoding="utf-8") as f:
        f.write("\n".join(english[int(0.8*len(english)):int(0.9*len(english))]))
    with codecs.open(os.path.join(data_folder, "val.ta"), "w", encoding="utf-8") as f:
        f.write("\n".join(tamil[int(0.8*len(tamil)):int(0.9*len(tamil))]))
                
    with codecs.open(os.path.join(data_folder, "test.en"), "w", encoding="utf-8") as f:
        f.write("\n".join(english[int(0.9*len(english)):]))
    with codecs.open(os.path.join(data_folder, "test.ta"), "w", encoding="utf-8") as f:
        f.write("\n".join(tamil[int(0.9*len(tamil)):]))
    

    print("\n...DONE!\n")
    

def get_positional_encoding(d_model, max_length=100):
    """
    Computes positional encoding as defined in the paper.

    :param d_model: size of vectors throughout the transformer model
    :param max_length: maximum sequence length up to which positional encodings must be calculated
    :return: positional encoding, a tensor of size (1, max_length, d_model)
    """
    positional_encoding = torch.zeros((max_length, d_model))  # (max_length, d_model)
    for i in range(max_length):
        for j in range(d_model):
            if j % 2 == 0:
                positional_encoding[i, j] = math.sin(i / math.pow(10000, j / d_model))
            else:
                positional_encoding[i, j] = math.cos(i / math.pow(10000, (j - 1) / d_model))

    positional_encoding = positional_encoding.unsqueeze(0)  # (1, max_length, d_model)

    return positional_encoding


def get_lr(step, d_model, warmup_steps):
    """
    The LR schedule. This version below is twice the definition in the paper, as used in the official T2T repository.

    :param step: training step number
    :param d_model: size of vectors throughout the transformer model
    :param warmup_steps: number of warmup steps where learning rate is increased linearly; twice the value in the paper, as in the official T2T repo
    :return: updated learning rate
    """
    lr = 2. * math.pow(d_model, -0.5) * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))

    return lr


def save_checkpoint(epoch, model, optimizer, prefix=''):
    """
    Checkpoint saver. Each save overwrites previous save.

    :param epoch: epoch number (0-indexed)
    :param model: transformer model
    :param optimizer: optimized
    :param prefix: checkpoint filename prefix
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = prefix + 'transformer_checkpoint.pth.tar'
    torch.save(state, filename)


def change_lr(optimizer, new_lr):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be changed
    :param new_lr: new learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

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
