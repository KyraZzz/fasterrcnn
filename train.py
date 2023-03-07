from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

if __init__ == "__main__":
    now = datetime.datetime.now()
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=2,
        verbose=True,
        mode='min'
    )

    # define the model
    num_classes = 43
    lr = 1e-3
    num_epochs = 100
    model = FasterRCNN(num_classes, lr)
    # define the log file
    logger = pl_loggers.TensorBoardLogger(save_dir = f'./logs/{now.month}-{now.day}/FasterRCNN/')
    # define the trainer
    trainer = pl.Trainer(max_epochs=num_epochs, accelerator="gpu", devices=1, callbacks= [early_stop_callback], logger=logger)
    # fit the model
    trainer.fit(model, train_dataloader, val_dataloader)

