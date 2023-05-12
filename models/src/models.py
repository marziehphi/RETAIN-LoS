from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from metrics import print_metrics_regression, print_metrics_mortality
from utils import BatchNorm1d
from utils import MSELoss, MSLELoss


def remove_padding(y, mask, device):
    """
        Filters out padding from tensor of predictions or labels
        Args:
            y: tensor of los predictions or labels
            mask (bool_type): tensor showing which values are padding (0) and which are data (1)
    """
    # note it's fine to call .cpu() on a tensor already on the cpu
    y = y.where(mask, torch.tensor(float('nan')).to(device=device)).flatten().detach().cpu().numpy()
    y = y[~np.isnan(y)]
    return y


class LiteLSTM(pl.LightningModule):
    def __init__(
            self,
            config,
            F=None,
            D=None,
            no_flat_features=None,
            mort_pred_time=24,
            time_before_pred=5,
            weight_decay=0.01,
            learning_rate=2e-5,
            adam_epsilon=1e-6,
    ):
        super().__init__()

        self.save_hyperparameters()

        if not config.channel_wise:
            self.lstm = nn.LSTM(
                input_size=(2 * F + 2),
                hidden_size=config.n_units,
                num_layers=config.n_layers,
                bidirectional=config.bidirectional,
                dropout=config.dropout_rate,
            )
        else:
            self.lstm = nn.ModuleList([
                nn.LSTM(
                    input_size=2,
                    hidden_size=config.n_units,
                    num_layers=config.n_layers,
                    bidirectional=config.bidirectional,
                    dropout=config.dropout_rate,
                ) for i in range(F)
            ])

        # input shape: B * D
        # output shape: B * diagnosis_size
        self.diagnosis_encoder = nn.Linear(in_features=D, out_features=config.diagnosis_size)

        # input shape: B * diagnosis_size
        # self.bn_diagnosis_encoder = nn.BatchNorm1d(num_features=config.diagnosis_size)
        self.bn_diagnosis_encoder = BatchNorm1d(num_features=config.diagnosis_size, momentum=config.momentum)

        # input shape: (B * T) * (n_units + diagnosis_size + no_flat_features)
        # output shape: (B * T) * last_linear_size
        channel_wise = F if config.channel_wise else 1
        input_size = config.n_units * config.n_dir * channel_wise + config.diagnosis_size + no_flat_features
        self.point_los = nn.Linear(in_features=input_size, out_features=config.last_linear_size)
        self.point_mort = nn.Linear(in_features=input_size, out_features=config.last_linear_size)

        # input shape: (B * T) * last_linear_size
        # self.bn_point_last_los = nn.BatchNorm1d(num_features=config.last_linear_size)
        # self.bn_point_last_mort = nn.BatchNorm1d(num_features=config.last_linear_size)
        self.bn_point_last_los = BatchNorm1d(num_features=config.last_linear_size, momentum=config.momentum)
        self.bn_point_last_mort = BatchNorm1d(num_features=config.last_linear_size, momentum=config.momentum)

        # input shape: (B * T) * last_linear_size
        # output shape: (B * T) * 1
        self.point_final_los = nn.Linear(in_features=config.last_linear_size, out_features=1)
        self.point_final_mort = nn.Linear(in_features=config.last_linear_size, out_features=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.hard_tanh = nn.Hardtanh(min_val=1 / 48, max_val=100)  # keep the end predictions between half an hour and 100 days
        self.dropout = nn.Dropout(config.dropout_rate)

        self.apply(self.init_weights)

        # self.remove_padding = lambda y, mask: remove_padding(y, mask, device=self.device)
        # self.remove_none = lambda x: tuple(xi for xi in x if xi is not None)

    def remove_padding(self, y, mask):
        return remove_padding(y, mask, device=self.device)

    def remove_none(self, x):
        return tuple(xi for xi in x if xi is not None)

    def init_weights(self, m):
        if isinstance(m, nn.LSTM):
            nn.init.xavier_uniform_(m.weight_ih_l0)
            nn.init.orthogonal_(m.weight_hh_l0)
            for names in m._all_weights:
                for name in filter(lambda n: 'bias' in n, names):
                    bias = getattr(m, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)
        return

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        h0, c0 = (torch.zeros(self.hparams.config.n_layers * self.hparams.config.n_dir, self.hparams.B, self.hparams.config.n_units),
                  torch.zeros(self.hparams.config.n_layers * self.hparams.config.n_dir, self.hparams.B, self.hparams.config.n_units))

        return h0, c0

    def setup(self, stage=None):
        """
        Handling the one-time task in a specified stage
        Args:
            stage (str): either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        """
        if stage != "fit":
            return

    def forward(
            self,
            time_series,
            diagnoses,
            flat
    ):
        """
        Feed data into our model.

        The timeseries data will be of dimensions B * (2F + 2) * T where:
          B is the batch size
          F is the number of features for convolution (N.B. we start with 2F because there are corresponding mask features)
          T is the number of timepoints
          The other 2 features represent the sequence number and the hour in the day

        The diagnoses data will be of dimensions B * D where:
          D is the number of diagnoses

        The flat data will be of dimensions B * no_flat_features
        """
        # print(f"time_series: {time_series.shape}")
        # print(f"diagnoses: {diagnoses.shape}")
        # print(f"flat: {flat.shape}")
        time_before_pred = self.hparams.time_before_pred
        B, _, T = time_series.shape
        x = time_series

        if not self.hparams.config.channel_wise:
            # the lstm expects (seq_len, batch, input_size)
            # N.B. the default hidden state is zeros so we don't need to specify it
            lstm_output, hidden = self.lstm(x.permute(2, 0, 1))  # T * B * hidden_size
            # print(lstm_output.shape, len(hidden), [h.shape for h in hidden])
        else:
            # take time and hour fields as they are not useful when processed on their own (they go up linearly. They were also taken out for temporal convolution so the comparison is fair)
            x = torch.split(x[:, 1:-1, :], self.hparams.F, dim=1)  # tuple ((B * F * T), (B * F * T))
            x = torch.stack(x, dim=2)  # B * F * 2 * T
            lstm_output = None
            for i in range(self.hparams.F):
                lstm, hidden = self.lstm[i](x[:, i, :, :].permute(2, 0, 1))
                lstm_output = torch.cat(self.remove_none((lstm_output, lstm)), dim=2)

        x = self.relu(self.dropout(lstm_output.permute(1, 2, 0)))
        # print(x.shape)

        # note that we cut off at time_before_pred hours here because the model is only valid from time_before_pred hours onwards
        diagnoses_enc = self.relu(self.dropout(self.bn_diagnosis_encoder(self.diagnosis_encoder(diagnoses))))  # B * diagnosis_size
        combined_features = torch.cat(
            (flat.repeat_interleave(T - time_before_pred, dim=0),  # (B * (T - time_before_pred)) * no_flat_features
             diagnoses_enc.repeat_interleave(T - time_before_pred, dim=0),  # (B * (T - time_before_pred)) * diagnosis_size
             x[:, :, time_before_pred:].permute(0, 2, 1).contiguous().view(B * (T - time_before_pred), -1)
             ), dim=1)
        # print(diagnoses_enc.shape)
        # print("combined_features", combined_features.shape)

        # print("layer", self.point_los)
        # print("last_point_los 1", last_point_los.shape)
        last_point_los = self.relu(self.dropout(self.bn_point_last_los(self.point_los(combined_features))))
        # print("last_point_los 2", last_point_los.shape)

        last_point_mort = self.relu(self.dropout(self.bn_point_last_mort(self.point_mort(combined_features))))
        # print(last_point_mort.shape)

        if self.hparams.config.no_exp:
            los_predictions = self.hard_tanh(self.point_final_los(last_point_los).view(B, T - time_before_pred))  # B * (T - time_before_pred)
        else:
            los_predictions = self.hard_tanh(torch.exp(self.point_final_los(last_point_los).view(B, T - time_before_pred)))  # B * (T - time_before_pred)

        mort_predictions = self.sigmoid(self.point_final_mort(last_point_mort).view(B, T - time_before_pred))  # B * (T - time_before_pred)

        return los_predictions, mort_predictions

    def configure_optimizers(self):
        """ Configure the optimizer and lr scheduler"""
        optimizer = torch.optim.Adam(self.parameters(), eps=self.hparams.adam_epsilon, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", verbose=True)
        scheduler = {"scheduler": scheduler, "interval": "step", "monitor": "train_loss"}
        return [optimizer]

    def loss_fn(
            self,
            y_hat_los,
            y_hat_mort,
            y_los,
            y_mort,
            mask,
            seq_lengths,
            sum_losses,
            loss_type
    ):

        if self.hparams.config.task == "mortality":
            loss = nn.BCELoss()(y_hat_mort, y_mort) * self.hparams.config.alpha
        else:
            if loss_type == "msle":
                _loss = MSLELoss()(y_hat_los, y_los, mask.type(torch.BoolTensor).to(self.device), seq_lengths, sum_losses)
            elif loss_type == "mse":
                _loss = MSELoss()(y_hat_los, y_los, mask.type(torch.BoolTensor).to(self.device), seq_lengths, sum_losses)

            if self.hparams.config.task == "LoS":
                loss = _loss

            if self.hparams.config.task == "multitask":
                loss = _loss + nn.BCELoss()(y_hat_mort, y_mort) * self.hparams.config.alpha

        return loss

    def training_step(self, batch, batch_index):
        """ Training process """
        time_series, mask, diagnoses, flat, los_labels, mort_labels, seq_lengths = batch
        y_hat_los, y_hat_mort = self(time_series, diagnoses, flat)
        loss = self.loss_fn(
            y_hat_los,
            y_hat_mort,
            los_labels,
            mort_labels,
            mask,
            seq_lengths,
            self.hparams.config.sum_losses,
            self.hparams.config.loss_type
        )

        output_dict = {"loss": loss}

        _y_hat_los, _y_los = [], []
        if self.hparams.config.task in ["LoS", "multitask"]:
            _y_hat_los = self.remove_padding(y_hat_los, mask.type(torch.BoolTensor).to(self.device))
            _y_los = self.remove_padding(los_labels, mask.type(torch.BoolTensor).to(self.device))

            output_dict["_y_hat_los"] = _y_hat_los
            output_dict["_y_los"] = _y_los

        _y_hat_mort, _y_mort = [], []
        mort_pred_time = self.hparams.mort_pred_time
        if self.hparams.config.task in ["mortality", "multitask"] and mort_labels.shape[1] >= mort_pred_time:
            _y_hat_mort = self.remove_padding(y_hat_mort[:, mort_pred_time], mask.type(torch.BoolTensor).to(self.device)[:, mort_pred_time])
            _y_mort = self.remove_padding(mort_labels[:, mort_pred_time], mask.type(torch.BoolTensor).to(self.device)[:, mort_pred_time])
            output_dict["_y_hat_mort"] = _y_hat_mort
            output_dict["_y_mort"] = _y_mort

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=time_series.size(0))
        return output_dict

    def training_epoch_end(self, outputs):
        _y_hat_los, _y_los = [], []
        _y_hat_mort, _y_mort = [], []
        for output in outputs:
            if "_y_hat_los" in output:
                _y_hat_los.append(output["_y_hat_los"])

            if "_y_los" in output:
                _y_los.append(output["_y_los"])

            if "_y_hat_mort" in output:
                _y_hat_mort.append(output["_y_hat_mort"])

            if "_y_mort" in output:
                _y_mort.append(output["_y_mort"])

        log_dict = {}

        if self.hparams.config.task in ["LoS", "multitask"]:
            _y_hat_los = np.concatenate(_y_hat_los, axis=0)
            _y_los = np.concatenate(_y_los, axis=0)

            los_metrics_list = print_metrics_regression(_y_los, _y_hat_los)
            for metric_name, metric in zip(['mad', 'mse', 'mape', 'msle', 'r2', 'kappa'], los_metrics_list):
                log_dict[f"{metric_name}"] = metric

        if self.hparams.config.task in ["mortality", "multitask"]:
            _y_hat_mort = np.concatenate(_y_hat_mort, axis=0)
            _y_mort = np.concatenate(_y_mort, axis=0)

            mort_metrics_list = print_metrics_mortality(_y_mort, _y_hat_mort)
            for metric_name, metric in zip(['acc', 'prec0', 'prec1', 'rec0', 'rec1', 'auroc', 'auprc', 'f1macro'], mort_metrics_list):
                log_dict[f"{metric_name}"] = metric

        for key, value in log_dict.items():
            self.log(f"train_{key}_epoch", value, on_step=False)

    def validation_step(self, batch, batch_index):
        """ Validation process """
        time_series, mask, diagnoses, flat, los_labels, mort_labels, seq_lengths = batch
        y_hat_los, y_hat_mort = self(time_series, diagnoses, flat)
        loss = self.loss_fn(
            y_hat_los,
            y_hat_mort,
            los_labels,
            mort_labels,
            mask,
            seq_lengths,
            self.hparams.config.sum_losses,
            self.hparams.config.loss_type
        )

        output_dict = {"loss": loss}

        _y_hat_los, _y_los = [], []
        if self.hparams.config.task in ["LoS", "multitask"]:
            _y_hat_los = self.remove_padding(y_hat_los, mask.type(torch.BoolTensor).to(self.device))
            _y_los = self.remove_padding(los_labels, mask.type(torch.BoolTensor).to(self.device))

            output_dict["_y_hat_los"] = _y_hat_los
            output_dict["_y_los"] = _y_los

        _y_hat_mort, _y_mort = [], []
        mort_pred_time = self.hparams.mort_pred_time
        if self.hparams.config.task in ["mortality", "multitask"] and mort_labels.shape[1] >= mort_pred_time:
            _y_hat_mort = self.remove_padding(y_hat_mort[:, mort_pred_time], mask.type(torch.BoolTensor).to(self.device)[:, mort_pred_time])
            _y_mort = self.remove_padding(mort_labels[:, mort_pred_time], mask.type(torch.BoolTensor).to(self.device)[:, mort_pred_time])
            output_dict["_y_hat_mort"] = _y_hat_mort
            output_dict["_y_mort"] = _y_mort

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=time_series.size(0))
        return output_dict

    def validation_epoch_end(self, outputs):
        _y_hat_los, _y_los = [], []
        _y_hat_mort, _y_mort = [], []
        for output in outputs:
            if "_y_hat_los" in output:
                _y_hat_los.append(output["_y_hat_los"])

            if "_y_los" in output:
                _y_los.append(output["_y_los"])

            if "_y_hat_mort" in output:
                _y_hat_mort.append(output["_y_hat_mort"])

            if "_y_mort" in output:
                _y_mort.append(output["_y_mort"])

        log_dict = {}

        if self.hparams.config.task in ["LoS", "multitask"]:
            _y_hat_los = np.concatenate(_y_hat_los, axis=0)
            _y_los = np.concatenate(_y_los, axis=0)

            los_metrics_list = print_metrics_regression(_y_los, _y_hat_los)
            for metric_name, metric in zip(['mad', 'mse', 'mape', 'msle', 'r2', 'kappa'], los_metrics_list):
                log_dict[f"{metric_name}"] = metric

        if self.hparams.config.task in ["mortality", "multitask"]:
            _y_hat_mort = np.concatenate(_y_hat_mort, axis=0)
            _y_mort = np.concatenate(_y_mort, axis=0)

            mort_metrics_list = print_metrics_mortality(_y_mort, _y_hat_mort)
            for metric_name, metric in zip(['acc', 'prec0', 'prec1', 'rec0', 'rec1', 'auroc', 'auprc', 'f1macro'], mort_metrics_list):
                log_dict[f"{metric_name}"] = metric

        for key, value in log_dict.items():
            self.log(f"val_{key}_epoch", value, on_step=False)

    def test_step(self, batch, batch_index):
        """ Testing process """
        time_series, mask, diagnoses, flat, los_labels, mort_labels, seq_lengths = batch
        y_hat_los, y_hat_mort = self(time_series, diagnoses, flat)
        loss = self.loss_fn(
            y_hat_los,
            y_hat_mort,
            los_labels,
            mort_labels,
            mask,
            seq_lengths,
            self.hparams.config.sum_losses,
            self.hparams.config.loss_type
        )

        output_dict = {"loss": loss}

        _y_hat_los, _y_los = [], []
        if self.hparams.config.task in ["LoS", "multitask"]:
            _y_hat_los = self.remove_padding(y_hat_los, mask.type(torch.BoolTensor).to(self.device))
            _y_los = self.remove_padding(los_labels, mask.type(torch.BoolTensor).to(self.device))

            output_dict["_y_hat_los"] = _y_hat_los
            output_dict["_y_los"] = _y_los

        _y_hat_mort, _y_mort = [], []
        mort_pred_time = self.hparams.mort_pred_time
        if self.hparams.config.task in ["mortality", "multitask"] and mort_labels.shape[1] >= mort_pred_time:
            _y_hat_mort = self.remove_padding(y_hat_mort[:, mort_pred_time], mask.type(torch.BoolTensor).to(self.device)[:, mort_pred_time])
            _y_mort = self.remove_padding(mort_labels[:, mort_pred_time], mask.type(torch.BoolTensor).to(self.device)[:, mort_pred_time])
            output_dict["_y_hat_mort"] = _y_hat_mort
            output_dict["_y_mort"] = _y_mort

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=time_series.size(0))
        return output_dict

    def test_epoch_end(self, outputs):
        _y_hat_los, _y_los = [], []
        _y_hat_mort, _y_mort = [], []
        for output in outputs:

            if "_y_hat_los" in output:
                _y_hat_los.append(output["_y_hat_los"])

            if "_y_los" in output:
                _y_los.append(output["_y_los"])

            if "_y_hat_mort" in output:
                _y_hat_mort.append(output["_y_hat_mort"])

            if "_y_mort" in output:
                _y_mort.append(output["_y_mort"])

        log_dict = {}

        if self.hparams.config.task in ["LoS", "multitask"]:
            _y_hat_los = np.concatenate(_y_hat_los, axis=0)
            _y_los = np.concatenate(_y_los, axis=0)

            los_metrics_list = print_metrics_regression(_y_los, _y_hat_los)
            for metric_name, metric in zip(['mad', 'mse', 'mape', 'msle', 'r2', 'kappa'], los_metrics_list):
                log_dict[f"{metric_name}"] = metric

        if self.hparams.config.task in ["mortality", "multitask"]:
            _y_hat_mort = np.concatenate(_y_hat_mort, axis=0)
            _y_mort = np.concatenate(_y_mort, axis=0)

            mort_metrics_list = print_metrics_mortality(_y_mort, _y_hat_mort)
            for metric_name, metric in zip(['acc', 'prec0', 'prec1', 'rec0', 'rec1', 'auroc', 'auprc', 'f1macro'], mort_metrics_list):
                log_dict[f"{metric_name}"] = metric

        for key, value in log_dict.items():
            self.log(f"test_{key}_epoch", value, on_step=False)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """ Predict process """
        time_series, mask, diagnoses, flat, los_labels, mort_labels, seq_lengths = batch
        return self(time_series, diagnoses, flat)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Defining specific model arguments """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.00129)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--adam_epsilon", type=float, default=1e-5)
        parser.add_argument("--time_before_pred", type=int, default=5)
        parser.add_argument("--mort_pred_time", type=int, default=24)
        return parser
