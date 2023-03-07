from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from datasets_util import get_dataloaders, get_datasets
from models_util import FasterRCNN, FreezeFasterRCNN
import argparse

def run(args):
    log_dir = os.path.expanduser('~') + "/fasterrcnn/tb_logs"
    logger = TensorBoardLogger(log_dir, name=args.task_name)
    now = datetime.datetime.now()
    checkpoint_callback = ModelCheckpoint(
        dirpath = f"checkpoints/{now.month}-{now.day}/{args.task_name}",
        filename = f"{args.task_name}-date={now.month}-{now.day}"+"-{epoch:02d}-{val_loss:.2f}",
        verbose = True,
        save_top_k = 1,
        monitor = "val_loss",
        mode = "min"
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=2,
        verbose=True,
        mode='min'
    )

    # get datasets
    meta_csv, train_csv, test_csv, num_classes = get_datasets()
    # get dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_csv, test_csv)

    # define the model
    model = None
    match args.model:
        case "FasterRCNN":
            model = FasterRCNN(num_classes, args.lr)
        case "Freeze":
            model = FreezeFasterRCNN(num_classes, args.freeze_depth, args.lr)
        case _:
            raise Exception("Model not supported")
    # define the trainer
    trainer = pl.Trainer(
        logger = logger,
        callbacks = [early_stop_callback,checkpoint_callback],
        max_epochs = args.num_epochs,
        log_every_n_steps = args.log_every_n_steps,
        accelerator = "gpu",
        devices = args.num_gpu_devices,
        strategy = "ddp",
    )
    # fit the model
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, verbose = True, dataloaders = test_dataloader)

if __init__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type = str, required = True, help = "Task name")
    parser.add_argument("--lr", type = float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type = int, default=100, help="Max number of epochs")
    parser.add_argument("--model", type = str, default="FasterRCNN", help="The model name")
    parser.add_argument("--freeze_depth", type = int, default=None, help="Freeze up to and including i th layer")
    parser.add_argument("--log_every_n_steps", type = int, default=20, help="Log every n steps for logging")
    parser.add_argument("--num_gpu_devices", type = int, default=1, help="Number of GPU")
    args = parser.parse_args()
    run(args)
    

