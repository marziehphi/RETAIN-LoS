import torch
from torch import nn


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_recurrent_layers=3):
        super(BiLSTMModel, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_recurrent_layers = n_recurrent_layers

        self.root_map = nn.Conv1d(input_size, embedding_size, kernel_size=1)
        self.bilstm_cell = nn.LSTM(embedding_size, hidden_size, n_recurrent_layers, batch_first=True, bidirectional=True)
        self.predictor = nn.Conv1d(hidden_size*2, 1, kernel_size=1)

        self.h_init = torch.zeros((self.n_recurrent_layers*2, 1, hidden_size))
        self.c_init = torch.zeros((self.n_recurrent_layers*2, 1, hidden_size))

    def tensors_to_cuda(self):
        self.h_init = self.h_init.cuda()
        self.c_init = self.c_init.cuda()

    def tensors_to_cpu(self):
        self.h_init = self.h_init.cpu()
        self.c_init = self.c_init.cpu()

    def forward(self, x, mode='train'):
        assert x.size(0) == 1, 'Only one example can be processed at once'
        assert mode in ['train', 'test'], 'Invalid mode. Must be "train" or "test"'

        # Initialize hidden layers
        self.h_init.fill_(0.0)
        self.c_init.fill_(0.0)

        x = x.permute(0, 2, 1)  # Interpret features as channels for 1D convolution
        v = self.root_map(x)
        v = v.permute(0, 2, 1)  # Move features back to last axis, for LSTM layer

        z, _ = self.bilstm_cell(v, (self.h_init, self.c_init))
        z = z.permute(0, 2, 1)  # Interpret features as channels for 1D convolution

        y = self.predictor(z).squeeze(1)  # Reshape to 1 x seq_length

        if mode == 'test':
            y = nn.ReLU()(y)

        return y


class LSTMModel(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_recurrent_layers=3):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_recurrent_layers = n_recurrent_layers

        self.root_map = nn.Conv1d(input_size, embedding_size, kernel_size=1)
        self.lstm_cell = nn.LSTM(embedding_size, hidden_size, n_recurrent_layers, batch_first=True)
        self.predictor = nn.Conv1d(hidden_size, 1, kernel_size=1)

        self.h_init = torch.zeros((self.n_recurrent_layers, 1, hidden_size))
        self.c_init = torch.zeros((self.n_recurrent_layers, 1, hidden_size))

    def tensors_to_cuda(self):
        self.h_init = self.h_init.cuda()
        self.c_init = self.c_init.cuda()

    def tensors_to_cpu(self):
        self.h_init = self.h_init.cpu()
        self.c_init = self.c_init.cpu()

    def forward(self, x, mode='train'):
        assert x.size(0) == 1, 'Only one example can be processed at once'
        assert mode in ['train', 'test'], 'Invalid mode. Must be "train" or "test"'

        # Initialize hidden layers
        self.h_init.fill_(0.0)
        self.c_init.fill_(0.0)

        x = x.permute(0, 2, 1)  # Interpret features as channels for 1D convolution
        v = self.root_map(x)
        v = v.permute(0, 2, 1)  # Move features back to last axis, for LSTM layer

        z, _ = self.lstm_cell(v, (self.h_init, self.c_init))
        z = z.permute(0, 2, 1)  # Interpret features as channels for 1D convolution

        y = self.predictor(z).squeeze(1)  # Reshape to 1 x seq_length

        if mode == 'test':
            y = nn.ReLU()(y)

        return y


class RETAINModel(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_recurrent_layers=3):
        super(RETAINModel, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_recurrent_layers = n_recurrent_layers

        self.root_map = nn.Conv1d(input_size, embedding_size, kernel_size=1)
        self.visit_attention_lstm = nn.LSTM(embedding_size, hidden_size, n_recurrent_layers, batch_first=True)
        self.visit_attention_map = nn.Conv1d(hidden_size, 1, kernel_size=1)
        self.feature_attention_lstm = nn.LSTM(embedding_size, hidden_size, n_recurrent_layers, batch_first=True)
        self.feature_attention_map = nn.Conv1d(hidden_size, embedding_size, kernel_size=1)
        self.predictor = nn.Conv1d(embedding_size, 1, kernel_size=1)

        self.alpha_h_init = torch.zeros((self.n_recurrent_layers, 1, hidden_size))
        self.alpha_c_init = torch.zeros((self.n_recurrent_layers, 1, hidden_size))
        self.beta_h_init = torch.zeros((self.n_recurrent_layers, 1, hidden_size))
        self.beta_c_init = torch.zeros((self.n_recurrent_layers, 1, hidden_size))

    def tensors_to_cuda(self):
        self.alpha_h_init = self.alpha_h_init.cuda()
        self.alpha_c_init = self.alpha_c_init.cuda()
        self.beta_h_init = self.beta_h_init.cuda()
        self.beta_c_init = self.beta_c_init.cuda()

    def tensors_to_cpu(self):
        self.alpha_h_init = self.alpha_h_init.cpu()
        self.alpha_c_init = self.alpha_c_init.cpu()
        self.beta_h_init = self.beta_h_init.cpu()
        self.beta_c_init = self.beta_c_init.cpu()

    def forward(self, x, interpret=False, mode='train', reverse_input=False):
        assert x.size(0) == 1, 'Only one example can be processed at once'
        assert mode in ['train', 'test'], 'Invalid mode. Must be "train" or "test"'

        # Initialize hidden layers
        self.alpha_h_init.fill_(0.0)
        self.alpha_c_init.fill_(0.0)
        self.beta_h_init.fill_(0.0)
        self.beta_c_init.fill_(0.0)

        x = x.permute(0, 2, 1)  # Interpret features as channels for 1D convolution
        v = self.root_map(x)
        v = v.permute(0, 2, 1)  # Move features back to last axis, for LSTM layer

        if reverse_input:
            v = torch.flip(v, (1,))

        alpha_z, _ = self.visit_attention_lstm(v, (self.alpha_h_init, self.alpha_c_init))
        beta_z, _ = self.feature_attention_lstm(v, (self.beta_h_init, self.beta_c_init))
        alpha_z = alpha_z.permute(0, 2, 1)  # Interpret features as channels for 1D convolution
        beta_z = beta_z.permute(0, 2, 1)  # Interpret features as channels for 1D convolution

        alpha = nn.Sigmoid()(self.visit_attention_map(alpha_z))
        beta = nn.Tanh()(self.feature_attention_map(beta_z))

        if reverse_input:
            v = torch.flip(v, (1,))
            alpha = torch.flip(alpha, (1,))
            beta = torch.flip(beta, (1,))

        v = v.permute(0, 2, 1)  # Make v compatible with 1D convolution again
        # v_weighted = torch.cumsum((v * beta) * alpha.repeat(1, self.embedding_size, 1), dim=2)
        v_weighted = (v * beta) * alpha.repeat(1, self.embedding_size, 1)

        y = self.predictor(v_weighted).squeeze(1)  # Reshape to 1 x seq_length

        if mode == 'test':
            y = nn.ReLU()(y)

        if not interpret:
            return y
        else:
            # Expand tensors to have the shape batch_size x embedding_size x input_size x seq_length
            beta_exp = beta.unsqueeze(2).repeat(1, 1, self.input_size, 1)
            alpha_exp = alpha.unsqueeze(1).unsqueeze(1).repeat(1, self.embedding_size, self.input_size, 1)
            W_emb_exp = self.root_map.weight.unsqueeze(0).repeat(x.size(0), 1, 1, x.size(-1))
            weight_pre = alpha_exp * beta_exp * W_emb_exp

            weights = torch.conv2d(weight_pre, self.predictor.weight.unsqueeze(-1)).squeeze().permute(0, 2, 1)
            return y, weights


class BiRETAINModel(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_recurrent_layers=3):
        super(BiRETAINModel, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_recurrent_layers = n_recurrent_layers

        self.root_map = nn.Conv1d(input_size, embedding_size, kernel_size=1)
        self.visit_attention_lstm = nn.LSTM(embedding_size, hidden_size, n_recurrent_layers, batch_first=True, bidirectional=True)
        self.visit_attention_map = nn.Conv1d(hidden_size*2, 1, kernel_size=1)
        self.feature_attention_lstm = nn.LSTM(embedding_size, hidden_size, n_recurrent_layers, batch_first=True, bidirectional=True)
        self.feature_attention_map = nn.Conv1d(hidden_size*2, embedding_size, kernel_size=1)
        self.predictor = nn.Conv1d(embedding_size, 1, kernel_size=1)

        self.alpha_h_init = torch.zeros((self.n_recurrent_layers*2, 1, hidden_size))
        self.alpha_c_init = torch.zeros((self.n_recurrent_layers*2, 1, hidden_size))
        self.beta_h_init = torch.zeros((self.n_recurrent_layers*2, 1, hidden_size))
        self.beta_c_init = torch.zeros((self.n_recurrent_layers*2, 1, hidden_size))

    def tensors_to_cuda(self):
        self.alpha_h_init = self.alpha_h_init.cuda()
        self.alpha_c_init = self.alpha_c_init.cuda()
        self.beta_h_init = self.beta_h_init.cuda()
        self.beta_c_init = self.beta_c_init.cuda()

    def tensors_to_cpu(self):
        self.alpha_h_init = self.alpha_h_init.cpu()
        self.alpha_c_init = self.alpha_c_init.cpu()
        self.beta_h_init = self.beta_h_init.cpu()
        self.beta_c_init = self.beta_c_init.cpu()

    def forward(self, x, interpret=False, mode='train'):
        assert x.size(0) == 1, 'Only one example can be processed at once'
        assert mode in ['train', 'test'], 'Invalid mode. Must be "train" or "test"'

        # Initialize hidden layers
        self.alpha_h_init.fill_(0.0)
        self.alpha_c_init.fill_(0.0)
        self.beta_h_init.fill_(0.0)
        self.beta_c_init.fill_(0.0)

        x = x.permute(0, 2, 1)  # Interpret features as channels for 1D convolution
        v = self.root_map(x)
        v = v.permute(0, 2, 1)  # Move features back to last axis, for LSTM layer

        alpha_z, _ = self.visit_attention_lstm(v, (self.alpha_h_init, self.alpha_c_init))
        beta_z, _ = self.feature_attention_lstm(v, (self.beta_h_init, self.beta_c_init))
        alpha_z = alpha_z.permute(0, 2, 1)  # Interpret features as channels for 1D convolution
        beta_z = beta_z.permute(0, 2, 1)  # Interpret features as channels for 1D convolution

        alpha = nn.Sigmoid()(self.visit_attention_map(alpha_z))
        beta = nn.Tanh()(self.feature_attention_map(beta_z))

        v = v.permute(0, 2, 1)  # Make v compatible with 1D convolution again
        # v_weighted = torch.cumsum((v * beta) * alpha.repeat(1, self.embedding_size, 1), dim=2)
        v_weighted = (v * beta) * alpha.repeat(1, self.embedding_size, 1)
        y = self.predictor(v_weighted).squeeze(1)  # Reshape to 1 x seq_length

        if mode == 'test':
            y = nn.ReLU()(y)

        if not interpret:
            return y
        else:
            # Expand tensors to have the shape batch_size x embedding_size x input_size x seq_length
            beta_exp = beta.unsqueeze(2).repeat(1, 1, self.input_size, 1)
            alpha_exp = alpha.unsqueeze(1).unsqueeze(1).repeat(1, self.embedding_size, self.input_size, 1)
            W_emb_exp = self.root_map.weight.unsqueeze(0).repeat(x.size(0), 1, 1, x.size(-1))
            weight_pre = alpha_exp * beta_exp * W_emb_exp

            weights = torch.conv2d(weight_pre, self.predictor.weight.unsqueeze(-1)).squeeze().permute(0, 2, 1)
            return y, weights



models_dict = {'bilstm': BiLSTMModel,'lstm': LSTMModel,'retain': RETAINModel,'biretain': BiRETAINModel}
