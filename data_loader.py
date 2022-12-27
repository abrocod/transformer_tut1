
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import dataset

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


def create_vocab():
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>']) 
    return vocab


def data_process(raw_text_iter: dataset.IterableDataset, tokenizer, vocab) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data: Tensor, batch_size: int, device) -> Tensor:
    """Divides the data into bsz (batch size) separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // batch_size
    data = data[:seq_len * batch_size]
    data = data.view(batch_size, seq_len).t().contiguous()
    return data.to(device)


def get_batch(source: Tensor, i: int, bptt: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    # import pdb; pdb.set_trace()
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


def get_data(split, batch_size, vocab, device):
    assert(split in ['train', 'test', 'valid'])
    tokenizer = get_tokenizer('basic_english')
    iter = WikiText2(split=split)
    data = data_process(iter, tokenizer=tokenizer, vocab=vocab)
    batched_data = batchify(data, batch_size, device=device)
    return batched_data


if __name__ == '__main__':    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vocab = create_vocab()
    tokenizer = get_tokenizer('basic_english')
    
    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again
    # train_iter, val_iter, test_iter = WikiText2()
    # train_data = data_process(train_iter, tokenizer=tokenizer, vocab=vocab)
    # val_data = data_process(val_iter, tokenizer=tokenizer, vocab=vocab)
    # test_data = data_process(test_iter, tokenizer=tokenizer, vocab=vocab)
    
    # batch_size = 20
    # eval_batch_size = 10
    # train_data = batchify(train_data, batch_size, device=device)  # shape [seq_len, batch_size]
    # val_data = batchify(val_data, eval_batch_size, device=device)
    # test_data = batchify(test_data, eval_batch_size, device=device)
    
    bptt = 35
    # print(get_batch(train_data, 100, bptt=bptt)[0].shape)
    # print(get_batch(train_data, 100, bptt=bptt)[1].shape)
    
    train_data = get_data(split='train', batch_size= 20, vocab=vocab, device=device)
    print(get_batch(train_data, 100, bptt=bptt)[0].shape)
    print(get_batch(train_data, 100, bptt=bptt)[1].shape)
    
    test_data = get_data(split='test', batch_size= 10, vocab=vocab, device=device)
    val_data = get_data(split='valid', batch_size= 10, vocab=vocab, device=device)
    print(get_batch(test_data, 100, bptt=bptt)[1].shape)
    print(get_batch(val_data, 100, bptt=bptt)[1].shape)
    
    data, targets = get_batch(train_data, i=10, bptt=bptt)