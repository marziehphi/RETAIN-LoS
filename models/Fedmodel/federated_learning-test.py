import torch
import numpy as np
from nn_utils import DataGenerator
import os
import pickle as pkl
from scipy.io import savemat
from sklearn.metrics import r2_score
import time
import pickle as pkl
import pandas as pd
from fed_utils import LocalUpdate
import warnings
import argparse
import progressbar
import copy
from fed_utils import LocalUpdate, FedAvg
warnings.filterwarnings('ignore')

torch.manual_seed(101)
np.random.seed(101)

parser = argparse.ArgumentParser(description='Code to test RNN and LSTM models')
#parser.add_argument('--path', help='Path to dataset', type=str, required=True)
parser.add_argument('--model_path', help='Path to model', type=str, required=True)
parser.add_argument('--no_cuda', dest='use_cuda', help='Flag to not use CUDA', action='store_false')
parser.add_argument('--reverse_input', help='Flag to reverse input', action='store_false')
parser.add_argument('--tag', help='Tag to add to results file name', type=str, default='')
parser.set_defaults(use_cuda=True)

args = parser.parse_args()
#assert os.path.exists(args.path), 'Path to dataset does not exist'
assert os.path.exists(args.model_path), 'Path to model does not exist'

args.batch_size = 1


path1 = '/workspace/los-prediction/los_v_0/eICU_preprocessing/eICU_data/eICU_Fed/mini_eicu_features_one'
path2 = '/workspace/los-prediction/los_v_0/eICU_preprocessing/eICU_data/eICU_Fed/mini_eicu_features_two'
path3 = '/workspace/los-prediction/los_v_0/eICU_preprocessing/eICU_data/eICU_Fed/mini_eicu_features_three'
path4 = '/workspace/los-prediction/los_v_0/eICU_preprocessing/eICU_data/eICU_Fed/mini_eicu_features_four'
data_generator1 = DataGenerator(path1, args.batch_size, mode='test', use_cuda=args.use_cuda)
data_generator2 = DataGenerator(path2, args.batch_size, mode='test', use_cuda=args.use_cuda)
data_generator3 = DataGenerator(path3, args.batch_size, mode='test', use_cuda=args.use_cuda)
data_generator4 = DataGenerator(path4, args.batch_size, mode='test', use_cuda=args.use_cuda)

data_generator = [data_generator1, data_generator2, data_generator3, data_generator4]

model = pkl.load(open(args.model_path, 'rb'))

if args.use_cuda:
    model = model.cuda()
    model.tensors_to_cuda()


lr_base_t0 = time.time()
print(time.localtime(lr_base_t0))
y_trues = []
y_preds = []
m = 2
client = np.random.choice(4, m, replace=False)
for idx in client:
    print('i = ', idx)
    dataset_test = data_generator[idx]
    y_trues = []
    y_preds = []
    for iter in range(dataset_test.steps_per_epoch):
        xs, ys = dataset_test.next()
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
            
    #y_trues = np.hstack(y_trues)
    y_trues_avg = sum(y_trues) / len(y_trues)
    y_preds_avg = sum(y_preds) / len(y_preds)
    y_preds = np.hstack(y_preds_avg)
    y_trues = np.hstack(y_trues_avg)
    mae = np.mean(np.abs(y_trues - y_preds))
    rmse = np.sqrt(np.mean((y_trues - y_preds)**2))
    r2 = r2_score(y_trues, y_preds)
    print("MAE: {}".format(mae))
    print("RMSE: {}".format(rmse))
    print("R2 score: {}".format(r2))
    lr_base_t1 = time.time()
        
#print(time.localtime(lr_base_t1))
#savemat(os.path.join('fedresults', model.__class__.__name__ + ('_reversed' if args.reverse_input else '') + ('_' + args.tag if len(args.tag) else '') + '_results.mat'),
#        {'y_trues': y_trues, 'y_preds': y_preds, 'mae': mae, 'rmse': rmse, 'r2_score': r2})
