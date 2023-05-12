import copy
import torch
from sklearn.metrics import r2_score


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


class LocalUpdate(object):
    def __init__(self, args, dataset):
        self.args = args
        self.steps_per_epoch = dataset.steps_per_epoch
        self.dataset = dataset

    def train(self, net):
        net.train()
        # Train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        batch_loss = []
        for iter in range(self.steps_per_epoch):
            xs, ys = self.dataset.next()
            y_preds = []
            loss = 0.0
            for x, y in zip(xs, ys):
                if self.args.reverse_input:
                    if self.args.model != 'retain':
                        x = torch.flip(x, (1,))
                        y_hat = net.forward(x)
                        y_hat = torch.flip(y_hat, (1,))
                    else:
                        y_hat = net.forward(x, reverse_input=True)
                else:
                    y_hat = net.forward(x)
                loss += torch.mean((y - y_hat) ** 2)  # MSE
                y_preds.append(y_hat)
            loss /= self.args.batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss)
        return net.state_dict(), sum(batch_loss)/len(batch_loss)
    def test(self, net):
        net.test()
        y_trues = []
        y_preds = []
        for iter in range(self.steps_per_epoch):
            xs, ys = self.dataset.next()
            y_trues.extend([y.squeeze().cpu().detach().numpy() for y in ys])
            for x, y in zip(xs, ys):
                if self.args.reverse_input:
                    if 'RETAIN' != model.__class__.__name__[:len('RETAIN')]:
                        x = torch.flip(x, (1,))
                        y_hat = net.forward(x)
                        y_hat = torch.flip(y_hat, (1,))
                    else:
                        y_hat = net.forward(x, reverse_input=True)
                else:
                    y_hat = net.forward(x)
                y_preds.append(y_hat.squeeze().cpu().detach().numpy())
                
        y_trues = np.hstack(y_trues)
        y_preds = np.hstack(y_preds)
        return net.state_dict(), sum(y_trues)/len(y_trues), sum(y_preds)/len(y_preds)
                    
            
    
    
