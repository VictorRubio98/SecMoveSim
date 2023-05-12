# encoding: utf-8
import torch
import numpy as np

from opacus.privacy_engine import forbid_accumulation_hook
import gc

def generate_samples(model, batch_size, seq_len, generated_num, output_file):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(
            batch_size, seq_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)


def generate_samples_to_mem(model, batch_size, seq_len, generated_num):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(
            batch_size, seq_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    return np.array(samples)


def pretrain_model(
        name,
        pre_epochs,
        model,
        data_iter,
        criterion,
        optimizer,
        batch_size,
        device=None):
    lloss = 0.
    for epoch in range(pre_epochs):
        loss = train_epoch(name, model, data_iter, criterion, optimizer, batch_size, device)
        print('Epoch [%d], loss: %f' % (epoch + 1, loss))
        if loss < 0.01 or 0 < lloss - loss < 0.01:
            print("early stop at epoch %d" % (epoch + 1))
            break
        
    gc.collect()
    torch.cuda.empty_cache()


def train_epoch(name, model, data_iter, criterion, optimizer, batch_size, device=None):
    total_loss = 0.
    if name == "G":
        tim = torch.LongTensor([i%24 for i in range(47)]).to(device)
        tim = tim.repeat(batch_size).reshape(batch_size, -1)
    for i, (data, labels) in enumerate(data_iter):
        data = torch.Tensor(data).to(dtype=torch.long, device=device)
        target = torch.Tensor(labels).to(dtype=torch.long, device=device)
        target = target.contiguous().view(-1)
        if name == "G":
            pred = model(data, tim)
        else:
            pred = model(data)
        loss = criterion(pred, target)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        data = data.to(device='cpu')
        target = target.to(device='cpu')
        gc.collect()
        torch.cuda.empty_cache()
    data_iter.reset()
    if name == 'G':
        tim = tim.to(device='cpu')

    return total_loss / (i + 1)

