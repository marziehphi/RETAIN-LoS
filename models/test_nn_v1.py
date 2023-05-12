import torch
import numpy as np
from nn_utils_v1 import DataGenerator
import os
import pickle as pkl
from scipy.io import savemat
from sklearn.metrics import r2_score
import argparse
import progressbar

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
import pickle as pkl

'''
test_nn_v1.py written to test on 4 chosen model for the _eicu_features.csv from _preprocess_eICU.py
Note: keep seed as 101 to get same results.
Note: keep ignore_time_series, use_first_record, use_last_record false.
Note: at the end you get patient ids because we shuffle during training step.
'''

torch.manual_seed(101)
np.random.seed(101)


parser = argparse.ArgumentParser(description='Code to test RNN and LSTM models')
parser.add_argument('--path', help='Path to dataset', type=str, required=True)
parser.add_argument('--model_path', help='Path to model', type=str, required=True)
parser.add_argument('--ignore_time_series', action='store_true', default=False) # predicting rLOS and True predicting LOS
parser.add_argument('--use_first_record', action='store_true', default=False)
parser.add_argument('--use_last_record', action='store_true', default=False)
parser.add_argument('--no_cuda', dest='use_cuda', help='Flag to not use CUDA', action='store_false')
parser.add_argument('--reverse_input', help='Flag to reverse input', action='store_true')
parser.add_argument('--tag', help='Tag to add to results file name', type=str, default='')
parser.set_defaults(use_cuda=True)

args = parser.parse_args()
assert os.path.exists(args.path), 'Path to dataset does not exist'
assert os.path.exists(args.model_path), 'Path to model does not exist'

widgets = [
            progressbar.ETA(),
            progressbar.Bar(),
            ]

args.batch_size = 1

#data_generator = DataGenerator(args.path, args.batch_size, mode='test', use_cuda=args.use_cuda)
data_generator = DataGenerator(args.path, args.batch_size, args.ignore_time_series, args.use_first_record, args.use_last_record, mode='test', use_cuda=args.use_cuda)

model = pkl.load(open(args.model_path, 'rb'))

if args.use_cuda:
    model = model.cuda()
    model.tensors_to_cuda()

with progressbar.ProgressBar(max_value=data_generator.steps_per_epoch, widgets=widgets) as bar:
    y_trues = []
    y_preds = []
    for i in range(data_generator.steps_per_epoch):
        xs, ys, ids = data_generator.next()
        print(ids)
        y_trues.extend([y.squeeze().cpu().detach().numpy() for y in ys])
        for x, y in zip(xs, ys):
            if args.reverse_input:
                if 'RETAIN' != model.__class__.__name__[:len('RETAIN')]:
                    x = torch.flip(x, (1,))
                    y_hat = model.forward(x)
                    y_hat = torch.flip(y_hat, (1,))
                else:
                    y_hat = model.forward(x, reverse_input=True)
            else:
                y_hat = model.forward(x)
            y_preds.append(y_hat.squeeze().cpu().detach().numpy())

        bar.update(i)

y_trues = np.hstack(y_trues)
y_preds = np.hstack(y_preds)

mae = np.mean(np.abs(y_trues - y_preds))
rmse = np.sqrt(np.mean((y_trues - y_preds)**2))
r2 = r2_score(y_trues, y_preds)

print("MAE: {}".format(mae))
print("RMSE: {}".format(rmse))
print("R2 score: {}".format(r2))


savemat(os.path.join('results', model.__class__.__name__ + ('_reversed' if args.reverse_input else '') + ('_ignore_time_series' if args.ignore_time_series else '') + ('_use_first_record' if args.use_first_record else '')+ ('_use_last_record' if args.use_last_record else '')+ ('_' + args.tag if len(args.tag) else '') + '_results.mat'),
        {'y_trues': y_trues, 'y_preds': y_preds, 'mae': mae, 'rmse': rmse, 'r2_score': r2})
