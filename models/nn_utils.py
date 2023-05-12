import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import time
import pickle as pkl


def save_model(model, save_dir, hyperparam_dict):
    filename = model.__class__.__name__
    for key, value in hyperparam_dict.items():
        filename = filename + '_' + key + '_' + str(value)

    t = time.localtime()
    filename = filename + '_' + str(t.tm_mday) + '_' + str(t.tm_mon) + '_' + str(t.tm_hour) + '_' + str(t.tm_min) + '.pkl'
    save_path = os.path.join(save_dir, filename)

    model = model.cpu()
    if 'tensors_to_cpu' in dir(model):
        model.tensors_to_cpu()

    pkl.dump(model, open(save_path, 'wb'))

    print("Model written to path: " + save_path)


class TargetEncoder():
    def __init__(self):
        self.category_maps = {}
        return

    def keys(self):
        return self.category_maps.keys()

    def fit(self, X, y, keys):
        if type(keys) != list:
            keys = [keys]

        for key in keys:
            print("Fitting column {}".format(key))
            category_map = {}
            for category, group in X.groupby(key, as_index=False):
                category_map[category] = y.loc[y.index.isin(group.index)].mean()
            category_map[''] = y.mean()
            self.category_maps[key] = category_map

    def transform(self, X):
        retX = X.copy()
        for key in retX.keys():
            if key in self.category_maps:
                retX[key] = retX[key].map(lambda x: self.category_maps[key][x] if x in self.category_maps[key] else self.category_maps[key][''])

        return retX


class DataGenerator(object):
    def __init__(self, path, batch_size=1, mode='train', use_cuda=True):
        assert mode in ['train', 'test'], 'Invalid mode. Must be "train" or "test"'

        self.path = path
        self.batch_size = batch_size
        self.mode = mode
        self.use_cuda = use_cuda

        df = pd.read_csv(path)
        df.drop(columns=['unitdischargeoffset', 'uniquepid', 'hospitaldischargestatus', 'unitdischargestatus'], inplace=True)
        df.set_index('patientunitstayid', inplace=True)
        self.y = df['rlos']
        self.X = df.drop(columns=['rlos'])
        del df

        self.stayids = self.X.index.unique()
        train_ids, test_ids = train_test_split(self.stayids, test_size=0.2, random_state=0)
        self.stayids = train_ids if mode == 'train' else test_ids
        self.n_ids = len(self.stayids)

        self.X = self.X.loc[self.stayids]
        self.y = self.y.loc[self.stayids]

        if not os.path.exists('models'):
            os.mkdir('models')

        encoder_path = os.path.join('models', 'targetencoder.pkl')
        if os.path.exists(encoder_path):
            encoder = pkl.load(open(encoder_path, 'rb'))
        else:
            if mode == 'test':
                print('Encoder not found')
                return

            print('Fitting TargetEncoder')
            encoder = TargetEncoder()
            encoder.fit(self.X, self.y, ['apacheadmissiondx', 'ethnicity', 'gender', 'GCS Total', 'Eyes', 'Motor', 'Verbal'])
            pkl.dump(encoder, open(encoder_path, 'wb'))

        print('Transforming using TargetEncoder')
        self.X = encoder.transform(self.X)

        scaler_path = os.path.join('models', 'minmaxscaler.pkl')
        if os.path.exists(scaler_path):
            scaler = pkl.load(open(scaler_path, 'rb'))
        else:
            if mode == 'test':
                print('Scaler not found')
                return

            print('Fitting MinMaxScaler')
            scaler = MinMaxScaler(feature_range=(-1, 1), copy=True)
            scaler.fit(self.X)
            pkl.dump(scaler, open(scaler_path, 'wb'))

        print('Transforming using MinMaxScaler')
        self.X[self.X.keys()] = scaler.transform(self.X)
        self.steps_per_epoch = self.n_ids//self.batch_size

        self.shuffle()

    def shuffle(self):
        self.inds = np.random.permutation(self.n_ids)
        self.step = 0

    def next(self):
        if self.step == self.steps_per_epoch:
            self.shuffle()

        ids = self.stayids[self.inds[self.step*self.batch_size: (self.step+1)*self.batch_size]]

        xs = []
        ys = []
        for train_id in ids:
            temp_x = self.X.loc[train_id].copy()
            temp_x = torch.from_numpy(temp_x.values).unsqueeze(0).float()
            temp_y = self.y.loc[train_id].copy()
            if len(temp_x.shape) == 2:
                temp_x = temp_x.unsqueeze(0)
                temp_y = torch.tensor([temp_y]).unsqueeze(0).unsqueeze(0).float()
            else:
                temp_y = torch.from_numpy(temp_y.values).unsqueeze(0).float()
            if self.use_cuda:
                temp_x = temp_x.cuda()
                temp_y = temp_y.cuda()
            xs.append(temp_x)
            ys.append(temp_y)

        self.step += 1
        return (xs, ys)


class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output, requires_grad=True)

    def close(self):
        self.hook.remove()
