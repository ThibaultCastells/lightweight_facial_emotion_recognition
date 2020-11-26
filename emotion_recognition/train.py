#!/usr/bin/env python
"""
To run this script:
python train.py --data_path "path/to/your/data/"
"""

from PIL import Image
import numpy as np
import os
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from datasets.faces_dataset import FacesDataset
from datasets.db_split import SplittedDataset
from archs.model import Model

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # params
    batch_size = 128
    lr = 0.01
    epochs = 120
    learning_rate_decay_start = 80
    learning_rate_decay_every = 5
    learning_rate_decay_rate = 0.9
    seed = 42
    shape = (50,50)

    # transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(shape[0]-4),
        transforms.RandomHorizontalFlip(),
        ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.CenterCrop(shape[0]-4),
        ToTensor(),
    ])

    # initialise model
    network = Model(num_classes=5).to(device)

    # initialise optimizer and loss
    optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.9, weight_decay=5e-3)
    criterion = nn.CrossEntropyLoss()

    # initialise data
    data = FacesDataset(True, args.data_path, shape=shape)

    n = len(data)
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)
    train_idx = idx[n//20:] # index for training samples
    val_idx = idx[:n//20] # index for evaluation samples
    print(f"Training samples: {len(train_idx)}")
    print(f"Evaluation samples: {len(val_idx)}")

    training_loader = DataLoader(SplittedDataset(data, train_idx, transform=train_transform), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(SplittedDataset(data, val_idx, transform=val_transform), batch_size=batch_size, shuffle=True)

    min_validation_loss = 10000

    for epoch in range(epochs):
        network.train()
        total = 0
        correct = 0
        total_train_loss = 0
        if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
            frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
            decay_factor = learning_rate_decay_rate ** frac
            current_lr = lr * decay_factor
            for group in optimizer.param_groups:
                group['lr'] = current_lr
        else:
            current_lr = lr

        print('learning_rate: %s' % str(current_lr))
        for i, (x_train, y_train) in enumerate(training_loader):
            optimizer.zero_grad()
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            y_predicted = network(x_train)
            loss = criterion(y_predicted, y_train)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(y_predicted.data, 1)
            total_train_loss += loss.data
            total += y_train.size(0)
            correct += predicted.eq(y_train.data).sum()
        accuracy = 100. * float(correct) / total
        print('Epoch [%d/%d] Training Loss: %.4f, Accuracy: %.4f' % (
            epoch + 1, epochs, total_train_loss / (i + 1), accuracy))

        network.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            total_validation_loss = 0
            for j, (x_val, y_val) in enumerate(validation_loader):
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                y_val_predicted = network(x_val)
                val_loss = criterion(y_val_predicted, y_val)
                _, predicted = torch.max(y_val_predicted.data, 1)
                total_validation_loss += val_loss.data
                total += y_val.size(0)
                correct += predicted.eq(y_val.data).sum()

            accuracy = 100. * float(correct) / total
            if total_validation_loss <= min_validation_loss:
                if epoch >= 10:
                    print('saving new model')
                    state = {'net': network.state_dict()}
                    torch.save(state, 'models/model_%d_%d.model' % (epoch + 1, accuracy))
                min_validation_loss = total_validation_loss

            print('Epoch [%d/%d] validation Loss: %.4f, Accuracy: %.4f' % (
                epoch + 1, epochs, total_validation_loss / (j + 1), accuracy))
