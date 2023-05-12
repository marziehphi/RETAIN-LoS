import torch
import numpy as np
from nn_utils import DataGenerator, save_model
from nn_models import BiLSTMModel
import os
import argparse
import time
import copy
from fed_utils import LocalUpdate, FedAvg
import warnings
from nn_models import models_dict
warnings.filterwarnings('ignore')

torch.manual_seed(101)
np.random.seed(101)

parser = argparse.ArgumentParser(description='Code to train RNN and intepretable RNN models')
parser.add_argument('--model', help='Model to train', type=str, required=True)
#parser.add_argument('--path', help='Path to dataset', type=str, required=True)
parser.add_argument('--epochs', help='Number of epochs for which to train the model', type=int, default=10)
parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
parser.add_argument('--reverse_input', help='Flag to reverse input', action='store_true')
parser.add_argument('--no_cuda', dest='use_cuda', help='Flag to not use CUDA', action='store_false')
parser.set_defaults(use_cuda=True)
#args = parser.parse_args(['--path', 'mini_eicu_features.csv'])
args = parser.parse_args()
#assert os.path.exists(args.path), 'Path to dataset does not exist'
assert args.model in models_dict, 'Invalid choice of model'

ModelClass = models_dict[args.model]

args.batch_size = 1
embedding_size = 40
hidden_size = 40

path1 = '/workspace/los-prediction/los_v_0/eICU_preprocessing/eICU_data/eICU_Fed/mini_eicu_features_one'
path2 = '/workspace/los-prediction/los_v_0/eICU_preprocessing/eICU_data/eICU_Fed/mini_eicu_features_two'
path3 = '/workspace/los-prediction/los_v_0/eICU_preprocessing/eICU_data/eICU_Fed/mini_eicu_features_three'
path4 = '/workspace/los-prediction/los_v_0/eICU_preprocessing/eICU_data/eICU_Fed/mini_eicu_features_four'
data_generator1 = DataGenerator(path1, args.batch_size, mode='train', use_cuda=args.use_cuda)
data_generator2 = DataGenerator(path2, args.batch_size, mode='train', use_cuda=args.use_cuda)
data_generator3 = DataGenerator(path3, args.batch_size, mode='train', use_cuda=args.use_cuda)
data_generator4 = DataGenerator(path4, args.batch_size, mode='train', use_cuda=args.use_cuda)

data_generator = [data_generator1, data_generator2, data_generator3, data_generator4]

#model_glob = BiLSTMModel(21, embedding_size, hidden_size)
model = ModelClass(21, embedding_size, hidden_size)

if args.use_cuda:
    model = model.cuda()
    model.tensors_to_cuda()

model.train()
w_glob = model.state_dict()

# Training
loss_train = []
lr_base_t0 = time.time()
print(time.localtime(lr_base_t0))
for iter in range(args.epochs):
    loss_locals = []
    w_locals = []
    m = 2
    client = np.random.choice(4, m, replace=False)
    for idx in client:
        print('i = ', idx)
        dataset_train = data_generator[idx]
        local = LocalUpdate(args=args, dataset=dataset_train)
        w, loss = local.train(net=copy.deepcopy(model))
        w_locals.append(copy.deepcopy(w))
        loss_locals.append(loss)

    w_glob = FedAvg(w_locals)
    model.load_state_dict(w_glob)

    loss_avg = sum(loss_locals) / len(loss_locals)
    print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
    loss_train.append(loss_avg)

lr_base_t1 = time.time()
print(time.localtime(lr_base_t1))
save_model(model, 'fedmodels', {'embedding_size': embedding_size, 'hidden_size': hidden_size, 'lr': args.lr, 'epochs': args.epochs, 'batch_size': args.batch_size, 'reversed': args.reverse_input})
