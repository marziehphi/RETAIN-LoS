import torch
import numpy as np
from nn_utils import DataGenerator, save_model
from nn_models import models_dict
import os
import argparse
import progressbar

torch.manual_seed(101)
np.random.seed(101)

parser = argparse.ArgumentParser(description='Code to train RNN and intepretable RNN models')
parser.add_argument('--path', help='Path to dataset', type=str, required=True)
parser.add_argument('--model', help='Model to train', type=str, required=True)
parser.add_argument('--epochs', help='Number of epochs for which to train the model', type=int, default=10)
parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
parser.add_argument('--no_cuda', dest='use_cuda', help='Flag to not use CUDA', action='store_false')
parser.add_argument('--reverse_input', help='Flag to reverse input', action='store_true')
parser.set_defaults(use_cuda=True)

args = parser.parse_args()
assert os.path.exists(args.path), 'Path to dataset does not exist'
assert args.model in models_dict, 'Invalid choice of model'

ModelClass = models_dict[args.model]

widgets = [
            progressbar.ETA(),
            progressbar.Bar(),
            ' ', progressbar.DynamicMessage('Error')
            ]

args.batch_size = 1
embedding_size = 40
hidden_size = 40

data_generator = DataGenerator(args.path, args.batch_size, mode='train', use_cuda=args.use_cuda)
model = ModelClass(21, embedding_size, hidden_size)
if args.use_cuda:
    model = model.cuda()
    model.tensors_to_cuda()

opt = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    tot_loss = 0.0

    print("Epoch {}/{}".format(epoch+1, args.epochs))
    with progressbar.ProgressBar(max_value=data_generator.steps_per_epoch, widgets=widgets) as bar:
        for i in range(data_generator.steps_per_epoch):
            xs, ys = data_generator.next()

            y_preds = []
            loss = 0.0
            for x, y in zip(xs, ys):
                if args.reverse_input:
                    if args.model != 'retain':
                        x = torch.flip(x, (1,))
                        y_hat = model.forward(x)
                        y_hat = torch.flip(y_hat, (1,))
                    else:
                        y_hat = model.forward(x, reverse_input=True)
                else:
                    y_hat = model.forward(x)
                loss += torch.mean((y - y_hat)**2)  # MSE
                y_preds.append(y_hat)
            loss /= args.batch_size

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot_loss += loss.detach().item()

            bar.update(i, Error=tot_loss/(i+1))

save_model(model, 'models', {'embedding_size': embedding_size, 'hidden_size': hidden_size, 'lr': args.lr, 'epochs': args.epochs, 'batch_size': args.batch_size, 'reversed': args.reverse_input})
