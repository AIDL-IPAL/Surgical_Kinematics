import os
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

def plot_losses(train_hist, test_hist, loss_history, meta_ep, tag, save_path):
    """
    train_hist / test_hist : list-like of per-epoch losses
    loss_history           : dataframe of loss histories
    meta_ep                : checkpoint epoch to show in the title
    tag                    : filename stem for the saved plot
    save_path              : root directory where plots are stored
    """
    # --- x-axis that starts at epoch 1 ---
    epochs = np.arange(0.5, len(train_hist) + 0.5)          # [1, 2, …, N]
    
    batch_epochs = np.arange(1, len(loss_history['batch_losses']) + 1)
    batchtoep = len(batch_epochs) / len(epochs)
    batch_epochs = batch_epochs / batchtoep  # now this works

    fig = plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_hist, label='Train Loss')
    plt.plot(epochs, loss_history['val_losses'],  label='Val  Loss')
    plt.plot(epochs, test_hist,  label='Test  Loss')
    
    plt.plot(batch_epochs, loss_history['batch_losses'], 'o', markersize=1.0, alpha=0.15, color='skyblue', label='Batch Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.xlim(0, len(epochs))                            # ensures the axis begins at X
    # Optional: show every epoch tick if you want discrete integers
    plt.xticks(epochs)
    plt.title(f'Loss History – Epoch {meta_ep+1}')
    plt.legend()

    plot_dir = os.path.join(save_path, 'training_plots')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'{tag}.png'), bbox_inches='tight')
    plt.show(block=False)
    # plt.pause(6)  # Pause to allow the plot to be displayed before closing
    plt.close(fig) 


def plot_batch_losses(batch_losses, epoch, rank, tag, save_path, me_loss=None, interval=1000):
        plt.figure()
        plt.plot(batch_losses, label='Batch Loss', alpha=0.5)
        if me_loss is not None:
            # Align ME loss with every 1000 batch calls
            x_values = np.arange((interval-1), (interval-1)+len(me_loss) * interval, interval)
            plt.plot(x_values, me_loss, label='ME Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.title(f'Batch/MiniEpoch Loss Curve - {tag}')
        plt.legend()
        plot_path = os.path.join(save_path, 'batch_plots')
        os.makedirs(plot_path, exist_ok=True)
        plot_filename = os.path.join(plot_path, f'batchloss_{tag}.png')
        plt.savefig(plot_filename)
        plt.show(block=False)
        # plt.pause(6)
        plt.close()
        # print(f"[Rank {rank}] Saved loss plot to {plot_filename}")
