from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import time
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

class FasterRCNN(pl.LightningModule):
  """ Train Faster R-CNN from scratch
  """
  def __init__(self, num_classes, lr=1e-3):
    super().__init__()
    # load the fasterrcnn model without pre-trained weights 
    self.model = fasterrcnn_resnet50_fpn(pretrain=False)
    # update the classifier layer for required number of classes
    in_features = self.model.roi_heads.box_predictor.cls_score.in_features
    self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # loss and mAP metrics
    self.metrics = CrossEntropyLoss()
    self.mAP = MeanAveragePrecision()
    # optimiser settings
    self.lr = lr
    self.step_size = 1
    self.save_hyperparameters()
  
  def forward(self, x, y=None):
    return self.model(x, y)
  
  def training_step(self, batch, batch_idx):
    x, y = batch
    # get model output: a dictionary of losses
    output = self.forward(x, y)
    # compute cumulative loss
    loss = torch.sum(torch.stack([loss for loss in output.values()],dim=0), dim=0)
    self.log('train_loss', loss, prog_bar=True, logger=True, sync_dist=True)
    return {'loss': loss}
  
  def validation_step(self, batch, batch_idx):
    x, y = batch
    with torch.no_grad():
      # compute loss, model only output loss when not in inference mode
      self.model.train()
      output = self.forward(x, y)
      loss = torch.sum(torch.stack([loss for loss in output.values()],dim=0), dim=0)
      self.log('val_loss', loss, prog_bar=True, logger=True, sync_dist=True)
    return {'val_loss': loss}
  
  def test_step(self, batch, batch_idx):
    x, y = batch
    # output contains List[Dict[Tensor]], one for each image
    #        including predicted bounding boxes, predicted labels and scores
    output = self.forward(x)
    scores = [torch.mean(out['scores']) for out in output]
    score = torch.mean(torch.stack(scores))
    self.log('test_score', score, prog_bar=True, logger=True, sync_dist=True)
    # compute mAP for each batch
    self.mAP.update(output, y)
    return {'test_score': score}
  
  def on_train_epoch_start(self):
    # start time of each epoch
    self.start_time = time.time()
  
  def test_epoch_end(self, outputs):
    mAP = self.mAP.compute()['map']
    self.log('mAP', mAP, logger=True, sync_dist=True)
    return {'mAP': mAP}
  
  def training_epoch_end(self, outputs):
    # compute the time taken for each epoch
    epoch_time = time.time() - self.start_time
    self.log('time_epoch', epoch_time, prog_bar=True, logger=True, sync_dist=True)
    # record the train epoch loss
    train_loss_epoch = outputs[-1]['loss']
    self.log('train_loss_epoch', train_loss_epoch, prog_bar=True, logger=True, sync_dist=True)
  
  def validation_epoch_end(self, outputs):
    # record the validation epoch loss
    val_score_epoch = outputs[-1]['val_loss']
    self.log('val_loss_epoch', val_score_epoch, prog_bar=True, logger=True, sync_dist=True)
  
  def configure_optimizers(self):
    # Adam optimiser with customised learning rate
    optimiser = Adam(self.parameters(), lr=self.lr)
    # linear learning rate scheduler to gradually increase learning rate
    scheduler = LinearLR(optimiser, self.step_size)
    return [optimiser], [scheduler]

# freeze the entire backbone network and train the classification layer from scratch on GTSRB dataset
class FreezeFasterRCNN(FasterRCNN):
  """ Freeze the backbone network up to and including K layers
      where K = freeze_depth, fine-tune the rest layers
  """
  def __init__(self, num_classes, freeze_depth=None, lr=1e-3):
    super().__init__(num_classes, lr)
    self.model = fasterrcnn_resnet50_fpn(pretrain=True)
    # freeze the backbone network 
    self.freeze_depth = freeze_depth
    if self.freeze_depth is not None:
      self.freeze_depth = min(freeze_depth, len(self.model.backbone.fpn.inner_blocks))
      for name, param in self.model.backbone.body.named_parameters():
        if f'layer{self.freeze_depth+1}' in name:
          break
        param.requires_grad = False
    # get the number of input features 
    in_features = self.model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    self.save_hyperparameters()