import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import time
import pickle as pkl

'''
nn_utils present the two main function:
save_model and DataGenerator for advance models for the second version of preprocessing eICU data.

Note: always ignore_time_series, use_first_record, use_last_record must be false. because we are using timeseries information 
in order to predict rLOS.

Note: we do not need targetencoder and scale part because it is done during preprocessing steps.

'''

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



class DataGenerator(object):
    def __init__(self, path, batch_size=1, ignore_time_series=False, use_first_record=False, use_last_record=False, mode='train', use_cuda=True):
        assert mode in ['train', 'test'], 'Invalid mode. Must be "train" or "test"'

        self.path = path
        self.batch_size = batch_size
        self.mode = mode
        self.use_cuda = use_cuda
        self.ignore_time_series = ignore_time_series
        self.use_first_record = use_first_record
        self.use_last_record = use_last_record

        df = pd.read_csv(path)
        if ignore_time_series:
            if use_first_record:
                df.set_index('patient', inplace=True)
                df_group = df.groupby('patient').first()
                df_group['LOS'] = (df_group['time']/24) + df_group['rlos']
                df_offsets = df.groupby('patient')['time'].size().reset_index()
                df_offsets.set_index('patient', inplace=True)
                df_offsets.index
                df = df_group.drop(['time', 'rlos'], axis=1).join(df_offsets, how='inner')
            elif use_last_record:
                df.set_index('patient', inplace=True)
                df_group = df.groupby('patient').last()
                df_group['LOS'] = (df_group['time']/24) + df_group['rlos']
                df = df_group.drop(['rlos'], axis=1)
            else:
                raise Exception('Must select first or last record')
                
            self.y = df['LOS']
            self.X = df.drop(columns=['LOS'])
        else:
            df.set_index('patient', inplace=True)
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

        #encoder_path = os.path.join('models', 'targetencoder.pkl')
        #if os.path.exists(encoder_path):
        #    encoder = pkl.load(open(encoder_path, 'rb'))
        #else:
        if mode == 'test':
            print('Encoder not found')
            return
        if mode == 'train':
            print('Encoder not found')
            return

            #print('Fitting TargetEncoder')
            #encoder = TargetEncoder()
            #encoder.fit(self.X, self.y, ['apacheadmissiondx', 'ethnicity', 'gender', 'GCS Total', 'Eyes', 'Motor', 'Verbal'])
           # pkl.dump(encoder, open(encoder_path, 'wb'))

        #print('Transforming using TargetEncoder')
        #self.X = encoder.transform(self.X)

        #scaler_path = os.path.join('models', 'minmaxscaler.pkl')
        #if os.path.exists(scaler_path):
        #    scaler = pkl.load(open(scaler_path, 'rb'))
        #else:
        if mode == 'test':
            print('Scaler not found')
            return
        if mode == 'train':
            print('Encoder not found')
            return

            #print('Fitting MinMaxScaler')
            #scaler = MinMaxScaler(feature_range=(-1, 1), copy=True)
            #scaler.fit(self.X)
            #pkl.dump(scaler, open(scaler_path, 'wb'))

        #print('Transforming using MinMaxScaler')
        #self.X[self.X.keys()] = scaler.transform(self.X)
        
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
        return (xs, ys, ids)

