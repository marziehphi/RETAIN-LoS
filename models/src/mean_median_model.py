import pandas as pd
import numpy as np
#import torch
from data_module import EICUDataModule
from data_module import EICUDataset
from models import remove_padding
from metrics import print_metrics_regression


path = 'YOUR_PATH'

train_datareader = EICUDataset(path + 'train')
test_datareader = EICUDataset(path + 'test')

train_batches = train_datareader.collate(batch = 512)
test_batches = test_datareader.collate(batch = 512)

train_y = np.array([])
test_y = np.array([])

for batch_idx, batch in enumerate(train_batches):
    padded, mask, diagnoses, flat, los_labels, mort_labels, seq_lengths = s
    train_y = np.append(train_y, remove_padding(los_labels, mask))

train_y = pd.DataFrame(train_y, columns=['true'])
mean_train = train_y.mean().values[0]
median_train = train_y.median().values[0]

for batch_idx, batch in enumerate(test_batches):
    padded, mask, diagnoses, flat, los_labels, mort_labels, seq_lengths = batch
    test_y = np.append(test_y, remove_padding(los_labels, mask))

test_y = pd.DataFrame(test_y, columns=['true'])
test_y['mean'] = mean_train
test_y['median'] = median_train

print('Total predictions:')
print('Using mean value of {}...'.format(mean_train))
print_metrics_regression(test_y['true'], test_y['mean'])
print('Using median value of {}...'.format(median_train))
print_metrics_regression(test_y['true'], test_y['median'])
