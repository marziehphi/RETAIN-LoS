import datetime
import os
from argparse import ArgumentParser
from pprint import pprint

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb
from data_module import EICUDataModule
from models import LiteLSTM
from utils import transform_dict

SEED = 101


class Config:

    def __init__(self, **kwargs):
        self.task = kwargs.get("task", None)
        self.hidden_size = kwargs.get("hidden_size", None)
        self.n_layers = kwargs.get("n_layers", None)
        self.bidirectional = kwargs.get("bidirectional", None)
        self.channel_wise = kwargs.get("channel_wise", None)
        self.dropout_rate = kwargs.get("dropout_rate", None)
        self.diagnosis_size = kwargs.get("diagnosis_size", None)
        self.last_linear_size = kwargs.get("last_linear_size", None)
        self.no_exp = kwargs.get("no_exp", None)
        self.sum_losses = kwargs.get("sum_losses", None)
        self.loss_type = kwargs.get("loss_type", None)
        self.alpha = kwargs.get("alpha", None)
        self.momentum = kwargs.get("momentum", None)
        self.n_units = self.hidden_size // 2 if self.bidirectional else self.hidden_size
        self.n_dir = 2 if self.bidirectional else 1

    def to_dict(self, expand: bool = True):
        return transform_dict(self.__dict__, expand)


def main():
    pl.seed_everything(SEED, workers=True)

    parser = ArgumentParser()
    parser.add_argument("--name", type=str)

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--labs_only", action="store_true")
    parser.add_argument("--no_labs", action="store_true")

    parser.add_argument("--task_name", type=str, default="LoS")
    parser.add_argument("--loss_type", type=str, default="msle")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--diagnosis_size", type=int, default=64)
    parser.add_argument("--last_linear_size", type=int, default=17)
    parser.add_argument("--alpha", type=int, default=100)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--momentum", type=float, default=0.01)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--channel_wise", action="store_false")
    parser.add_argument("--no_exp", action="store_true")
    parser.add_argument("--sum_losses", action="store_false")

    parser.add_argument("--es", action="store_true")
    parser.add_argument("--es_monitor", type=str, default="eval/loss")
    parser.add_argument("--es_min_delta", type=float, default=0.0)
    parser.add_argument("--es_patience", type=int, default=5)
    parser.add_argument("--es_mode", type=str, default="min")
    parser.add_argument("--es_verbose", action="store_true")

    parser.add_argument("--tb", action="store_true")

    parser.add_argument("--ckpt", action="store_true")
    parser.add_argument("--ckpt_save_top_k", type=int, default=1)
    parser.add_argument("--ckpt_verbose", action="store_true")
    parser.add_argument("--ckpt_monitor", type=str, default="val_loss")
    parser.add_argument("--ckpt_mode", type=str, default="min")

    parser.add_argument("--wandb_key", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LiteLSTM.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.wandb_key == "None" or args.wandb_key == "":
        args.wandb_key = None

    if args.wandb_project == "None" or args.wandb_project == "":
        args.wandb_project = None


    args.output_dir = os.path.join(args.output_dir, args.name)
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "ckpts")
    log_dir = os.path.join(args.output_dir, "logs")
    print(ckpt_dir)
    print(log_dir)

    # Args overview
    pprint(args.__dict__)

    # Setup wandb
    loggers = []
    if isinstance(args.wandb_key, str) and len(args.wandb_key) > 0:
        wandb.login(key=args.wandb_key, relogin=True)
        loggers += [WandbLogger(name=args.name, project=args.wandb_project)]

    data = EICUDataModule(
        data_path=args.data_path,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        labs_only=args.labs_only,
        no_labs=args.no_labs
    )
    data.prepare_data()
    data.setup(stage="fit")

    config = Config(**{
        "task": args.task_name,
        "hidden_size": args.hidden_size,
        "n_layers": args.n_layers,
        "n_units": 0,
        "bidirectional": args.bidirectional,
        "channel_wise": args.channel_wise,
        "dropout_rate": args.dropout_rate,
        "diagnosis_size": args.diagnosis_size,
        "last_linear_size": args.last_linear_size,
        "no_exp": args.no_exp,
        "sum_losses": args.sum_losses,
        "loss_type": args.loss_type,
        "alpha": args.alpha,
        "momentum": args.momentum,
    })

    pprint(config.to_dict())

    model = LiteLSTM(
        config=config,
        F=data.F,
        D=data.D,
        no_flat_features=data.no_flat_features
    )

    # Callbacks
    callbacks = [
        pl.callbacks.RichProgressBar(),
        pl.callbacks.LearningRateMonitor(logging_interval="step")
    ]

    if args.tb:
        loggers += [
            pl.loggers.TensorBoardLogger(
                save_dir=log_dir,
                version="version_" + datetime.datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
                name=args.name
            )
        ]

    if args.es:
        callbacks += [
            pl.callbacks.EarlyStopping(
                monitor=args.es_monitor,
                min_delta=args.es_min_delta,
                patience=args.es_patience,
                verbose=args.es_verbose,
                mode=args.es_mode
            )
        ]

    if args.ckpt:
        callbacks += [
            pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_dir,
                save_top_k=args.ckpt_save_top_k,
                verbose=args.ckpt_verbose,
                monitor=args.ckpt_monitor,
                mode=args.ckpt_mode
            )
        ]

    # print(callbacks)
    # raise

    # Training
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        logger=loggers if len(loggers) > 0 else True,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        precision=args.precision,
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=data)
    trainer.save_checkpoint(os.path.join(ckpt_dir, "last.ckpt"))

    # Testing
    trainer.test(model=model, datamodule=data)


if __name__ == '__main__':
    main()
