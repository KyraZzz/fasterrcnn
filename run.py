from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import argparse
import os
import datetime
from datasets_util import get_dataloaders, get_datasets
from models_util import ScratchFasterRCNN, FreezeFasterRCNN
from plot_utils import plot_metrics

def run(args):
    print(f"Parameter list: {chr(10)} \
    random seed: {args.random_seed}{chr(10)} \
    task name: {args.task_name}{chr(10)} \
    model name: {args.model}{chr(10)} \
    learning rate: {args.lr}{chr(10)} \
    do train: {args.do_train}{chr(10)} \
    do test: {args.do_test}{chr(10)} \
    checkpoint path: {args.ckpt_path}{chr(10)} \
    maximum epochs: {args.num_epochs}{chr(10)} \
    number of gpu devices: {args.num_gpu_devices}{chr(10)} \
    log every n steps: {args.log_every_n_steps}{chr(10)} \
    freeze depth: {args.freeze_depth}{chr(10)} \
    ")
    pl.seed_everything(args.random_seed)
    log_dir = os.path.expanduser('~') + "/fasterrcnn/tb_logs"
    logger = TensorBoardLogger(log_dir, name=args.task_name)
    now = datetime.datetime.now()
    checkpoint_callback = ModelCheckpoint(
        dirpath = f"checkpoints/{now.month}-{now.day}/{args.task_name}",
        filename = f"{args.task_name}-date={now.month}-{now.day}"+"-{epoch:02d}-{val_loss:.2f}",
        verbose = True,
        save_top_k = 1,
        monitor = "val_loss_epoch",
        mode = "min"
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss_epoch',
        min_delta=0.001,
        patience=2,
        verbose=True,
        mode='min'
    )

    # get datasets
    meta_csv, train_csv, test_csv, num_classes = get_datasets()
    # get dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(num_classes, train_csv, test_csv)

    # define the model
    model = None
    match args.model:
        case "FasterRCNN":
            if args.ckpt_path is not None:
                model = ScratchFasterRCNN.load_from_checkpoint(args.ckpt_path)
            else:
                model = ScratchFasterRCNN(num_classes, args.lr)
        case "Freeze":
            if args.ckpt_path is not None:
                model = FreezeFasterRCNN.load_from_checkpoint(args.ckpt_path)
            else:
                model = FreezeFasterRCNN(num_classes, args.freeze_depth, args.lr)
        case _:
            raise Exception("Model not supported")
    # define the trainer
    if args.num_gpu_devices > 1:
        trainer = pl.Trainer(
            logger = logger,
            callbacks = [early_stop_callback,checkpoint_callback],
            max_epochs = args.num_epochs,
            log_every_n_steps = args.log_every_n_steps,
            accelerator = "gpu",
            devices = args.num_gpu_devices,
            strategy = "ddp",
        )
    else:
        trainer = pl.Trainer(
            logger = logger,
            callbacks = [early_stop_callback,checkpoint_callback],
            max_epochs = args.num_epochs,
            log_every_n_steps = args.log_every_n_steps,
            accelerator = "gpu",
            devices = args.num_gpu_devices,
        )
    # fit the model
    if args.do_train:
        trainer.fit(model, train_dataloader, val_dataloader)
    if args.do_test:
        trainer.test(model, test_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type = int, default=42, help="Max number of epochs")
    parser.add_argument("--task_name", type = str, required = True, help = "Task name")
    parser.add_argument("--lr", type = float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type = int, default=10, help="Max number of epochs")
    parser.add_argument("--model", type = str, default="FasterRCNN", help="The model name")
    parser.add_argument("--freeze_depth", type = int, default=0, help="Freeze up i th layer")
    parser.add_argument("--log_every_n_steps", type = int, default=20, help="Log every n steps for logging")
    parser.add_argument("--num_gpu_devices", type = int, default=1, help="Number of GPU")
    parser.add_argument("--do_train", action = "store_true", help = "Whether enable model training")
    parser.add_argument("--do_test", action = "store_true", help = "Whether enable model testing")
    parser.add_argument("--ckpt_path", type = str, default = None, help = "Required for testing with checkpoint path")
    args = parser.parse_args()
    run(args)
    

