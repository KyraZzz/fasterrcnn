import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')
plt.rc('font', family='DejaVu Serif')

def plot_metrics(trainer, filename):
  """ Utility function to visualise the trainer logs
  """
  # gather the logs
  train_loss = [trainer.callback_metrics[f'train_loss_epoch_{i}'].detach().item() for i in range(trainer.current_epoch)]
  val_loss = [trainer.callback_metrics[f'val_loss_epoch_{i}'].detach().item() for i in range(trainer.current_epoch)]
  time_epoch = [trainer.callback_metrics[f'time_epoch_{i}'].detach().item() for i in range(trainer.current_epoch)]
  map_epoch = [trainer.callback_metrics[f'map_epoch_{i}'].detach().item() for i in range(trainer.current_epoch)]
  # plot the logs
  epochs = np.arange(trainer.current_epoch)
  fig, ax = plt.subplots(3,1,figsize=(6,6), sharex=True)
  ax[0].plot(epochs, train_loss, label="training loss")
  ax[0].plot(epochs, val_loss, label="validation loss")
  ax[1].plot(epochs, map_epoch, label="mAP")
  ax[2].plot(epochs, time_epoch, label="time")
  # anotate axis
  ax[2].set_xlabel('#epoch')
  ax[0].set_ylabel('Loss')
  ax[1].set_ylabel('mAP')
  ax[2].set_ylabel('time')
  # mark legend
  ax[0].legend()
  ax[1].legend()
  ax[2].legend()
  plt.savefig(f"{filename}.png")

def plot_metrics_comparison(trainer_list, trainer_name):
  train_loss = [[trainer.callback_metrics[f'train_loss_epoch_{i}'].detach().item() for i in range(trainer.current_epoch)] for trainer in trainer_list]
  val_loss = [[trainer.callback_metrics[f'val_loss_epoch_{i}'].detach().item() for i in range(trainer.current_epoch)] for trainer in trainer_list]
  time_epoch = [[trainer.callback_metrics[f'time_epoch_{i}'].detach().item() for i in range(trainer.current_epoch)] for trainer in trainer_list]
  map_epoch = [[trainer.callback_metrics[f'map_epoch_{i}'].detach().item() for i in range(trainer.current_epoch)] for trainer in trainer_list]
  epochs = [np.arange(trainer.current_epoch) for trainer in trainer_list]
  fig, ax = plt.subplots(4,1,figsize=(6,6), sharex=True)
  for idx, trainer_name in enumerate(trainer_name):
    ax[0].plot(epochs[idx], train_loss[idx], label=f"{trainer_name}")
    ax[1].plot(epochs[idx], val_loss[idx], label=f"{trainer_name}")
    ax[2].plot(epochs[idx], map_epoch[idx], label=f"{trainer_name}")
    ax[3].plot(epochs[idx], time_epoch[idx], label=f"{trainer_name}")
  ax[3].set_xlabel('#epoch')
  ax[0].set_ylabel('Train Loss')
  ax[1].set_ylabel('Validation Loss')
  ax[2].set_ylabel('mAP')
  ax[3].set_ylabel('time')
  ax[3].legend(loc='lower center', ncol=3, fancybox=True, bbox_to_anchor=(0.5, -0.9))
  plt.tight_layout()
  plt.show()