# metric3d_ddp_train.py
import os, sys
from pathlib import Path
import torch
from torch import amp
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import DataLoader, DistributedSampler
from datetime import datetime

# Add project root to path
try:
    script_path = Path(__file__).resolve()
    ROOT = script_path.parents[2]  # Adjust parent level if needed
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
except Exception as e:
    print(f"Warning: Could not automatically add project root to sys.path. Error details: {e}")
    print("Please ensure the project structure is as expected and __file__ is available, or set PYTHONPATH manually.")

    # Program execution will continue. Subsequent imports might fail if the path is not correctly set.


from scene3d.utils.dataprocess import DepthDataset, metric3d_unpad_and_scale, postprocess_depth
from utils.visualization import plot_losses, plot_batch_losses
from utils.hamlyn_intrinsics import read_intrinsics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F


def setup_ddp(rank, world_size):
    backend = 'nccl' if torch.cuda.is_available() and os.name != 'nt' else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def train_one_epoch(model, scale, train_loader, val_loader, test_loader, optimizer, criterion, rank, plot=True, epoch=0, loss_history_df=None, tag='', save_path='.'):
    model.train()
    total_loss = 0

    scaler = amp.GradScaler()

    # nonstandard_shape_count = 0
    target_shape = (480, 720)

    for batch in tqdm(train_loader, disable=rank != 0):
        rgb, depth_gt, mask, rgb_og, fname, pad_info, intrinsic, original_shapes = batch
        # for shape in original_shapes:
        #     if shape != target_shape:
        #         nonstandard_shape_count += 1

        rgb = rgb.to(rank)
        depth_gt = depth_gt.to(rank)
        mask = mask.to(rank)
        pad_info = [p for p in pad_info]  # Convert from [B, 4] tensor to list of 4-element tensors
        intrinsic = [k for k in intrinsic]

        with amp.autocast(device_type='cuda'):
            pred_depth, confidence, output_dict = model({'input': rgb})
            og_shape = depth_gt.shape[1:]
            pred_depth = metric3d_unpad_and_scale(pred_depth, pad_info, intrinsic, original_shapes)

            # Resize ground truth to match predicted shape
            depth_gt = F.interpolate(depth_gt.unsqueeze(1), size=pred_depth.shape[-2:], mode='bilinear', align_corners=False)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=pred_depth.shape[-2:], mode='nearest').bool()
            pred_depth = postprocess_depth(pred_depth, mask, scale=scale)
            depth_gt = (depth_gt * mask).squeeze(1)
            pred_depth = (pred_depth * mask).squeeze(1) # squeeze color channel dim
            loss = criterion(pred_depth, depth_gt)

            # --- Debug: Save plot instead of showing ---
            debug = True
            if debug:
                if len(loss_history_df['batch_losses']) % 500 == 0 and rank == 0:
                    pred_sample = pred_depth[0].detach().cpu().numpy()
                    gt_sample = depth_gt[0].detach().cpu().numpy()
                    rgb_s = rgb[0].detach().cpu().numpy()
                    from scene3d.utils.dataprocess import denormalize_rgb
                    rgb_sample, rgb_raw = denormalize_rgb(rgb_s)

                    # Create and save the debug figure
                    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
                    axs[0].imshow(rgb_sample)
                    axs[0].set_title('RGB Image')
                    axs[1].imshow(pred_sample, cmap='plasma')
                    axs[1].set_title('Mono. Predicted Depth')
                    axs[2].imshow(gt_sample, cmap='plasma')
                    axs[2].set_title('Stereo Depth Annotation')
                    plt.tight_layout()

                    # Save to disk
                    debug_dir = os.path.join(save_path, 'debug_plots')
                    os.makedirs(debug_dir, exist_ok=True)
                    debug_path = os.path.join(debug_dir, f'batch_{len(loss_history_df["batch_losses"])}.png')
                    plt.savefig(debug_path)
                    plt.close(fig)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loss_val = loss.item()
        total_loss += loss_val
        current_lr = optimizer.param_groups[0]['lr']
        loss_history_df['batch_losses'].append(loss_val)

        interval = 500 # loss plot interval
        if plot and rank == 0 and len(loss_history_df['batch_losses']) % interval == 0:
            tag_b = "epoch" + str(epoch) + "_miniepoch" + str(int(len(loss_history_df['batch_losses'])/interval))
            me_loss = np.mean(loss_history_df['batch_losses'][-interval:])
            loss_history_df['me_losses'].append(me_loss)
            plot_batch_losses(loss_history_df['batch_losses'], (epoch), rank, tag_b, save_path, me_loss=loss_history_df['me_losses'], interval=interval)
            print(f"Tag: {tag_b} | Loss: {np.mean(loss_history_df['batch_losses'][-interval:]):.4f}")
            # print(f"Non-standard shape count: {nonstandard_shape_count}")

    avg_loss = total_loss / len(train_loader)
    loss_history_df['train_losses'].append(avg_loss)

    
    
    # Validation & Test
    model.eval()
    
    print("Validation Inference:")
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, disable=rank != 0):
            rgb, depth_gt, mask, rgb_og, fname, pad_info, intrinsic, original_shapes = batch
            og_shape = depth_gt.shape[1:]
            rgb = rgb.to(rank)
            depth_gt = depth_gt.to(rank)
            mask = mask.to(rank)
            pad_info = [p for p in pad_info]  # Convert to list of 4-element tensors
            intrinsic = [k for k in intrinsic]

            with amp.autocast(device_type='cuda'):
                pred_depth, confidence, output_dict = model({'input': rgb})
                pred_depth = metric3d_unpad_and_scale(pred_depth, pad_info, intrinsic, original_shapes)
                depth_gt = F.interpolate(depth_gt.unsqueeze(1), size=pred_depth.shape[-2:], mode='bilinear', align_corners=False)
                mask = F.interpolate(mask.unsqueeze(1).float(), size=pred_depth.shape[-2:], mode='nearest').bool()
                pred_depth = postprocess_depth(pred_depth, mask, scale=scale)

                depth_gt = (depth_gt * mask).squeeze(1)
                pred_depth = (pred_depth * mask).squeeze(1) # squeeze color channel dim
            
                loss = criterion(pred_depth, depth_gt)
                
            val_loss += loss.item()
            loss_history_df['batch_val_losses'].append(loss.item())
            
            if plot and rank == 0 and len(loss_history_df['batch_val_losses']) % interval == 0:
                tag_b = tag + "val_loss_miniep_" + str(len(loss_history_df['batch_val_losses'])/interval)
                me_vloss = np.mean(loss_history_df['batch_val_losses'][-interval:])
                loss_history_df['me_val_losses'].append(me_vloss)
                # val_path = os.path.join(save_path, 'val_plot')
                # plot_batch_losses(batch_val_losses, (epoch), rank, tag_b, val_path, me_loss=me_vlosses, interval=interval)
                print(f"Mini-Epoch Val Loss: {np.mean(loss_history_df['batch_val_losses'][-interval:]):.4f}")

    avg_val_loss = val_loss / len(val_loader)    
    loss_history_df['val_losses'].append(avg_val_loss)

    # isolated testset inference
    print("Testset Inference:")
    test_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, disable=rank != 0):
            rgb, depth_gt, mask, rgb_og, fname, pad_info, intrinsic, original_shapes = batch
            og_shape = depth_gt.shape[1:]
            rgb = rgb.to(rank)
            depth_gt = depth_gt.to(rank)
            mask = mask.to(rank)
            pad_info = [p for p in pad_info]  # Convert to list of 4-element tensors
            intrinsic = [k for k in intrinsic]

            with amp.autocast(device_type='cuda'):
                pred_depth, confidence, output_dict = model({'input': rgb})
                pred_depth = metric3d_unpad_and_scale(pred_depth, pad_info, intrinsic, original_shapes)
                depth_gt = F.interpolate(depth_gt.unsqueeze(1), size=pred_depth.shape[-2:], mode='bilinear', align_corners=False)
                mask = F.interpolate(mask.unsqueeze(1).float(), size=pred_depth.shape[-2:], mode='nearest').bool()
                pred_depth = postprocess_depth(pred_depth, mask, scale=scale)

                depth_gt = (depth_gt * mask).squeeze(1)
                pred_depth = (pred_depth * mask).squeeze(1) # squeeze color channel dim
            
                loss = criterion(pred_depth, depth_gt)
                
            test_loss += loss.item()
            loss_history_df['batch_test_losses'].append(loss.item())
            
            if plot and rank == 0 and len(loss_history_df['batch_test_losses']) % interval == 0:
                tag_b = tag + "val_loss_miniep_" + str(len(loss_history_df['batch_test_losses'])/interval)
                me_tloss = np.mean(loss_history_df['batch_test_losses'][-interval:])
                loss_history_df['me_test_losses'].append(me_tloss)
                # test_path = os.path.join(save_path, 'val_plot')
                # plot_batch_losses(batch_test_losses, (epoch), rank, tag_b, test_path, me_loss=me_vlosses, interval=interval)
                print(f"Mini-Epoch Test Loss: {np.mean(loss_history_df['batch_test_losses'][-interval:]):.4f}")

    avg_test_loss = test_loss / len(test_loader)
    loss_history_df['test_losses'].append(avg_test_loss)

    
    
    if rank == 0:
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")
        print(f"[Epoch {epoch}] Val Loss: {avg_val_loss:.4f}")
        print(f"[Epoch {epoch}] Test Loss: {avg_test_loss:.4f}")

    return avg_loss, avg_test_loss, loss_history_df



def ddp_main(rank, world_size, args):
    if world_size > 1:
        setup_ddp(rank, world_size)
        
    model = torch.hub.load(args.model_path, args.model_type, pretrain=True, trust_repo=True)
    if args.model_ckpt and os.path.exists(args.model_ckpt):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        model.load_state_dict(torch.load(args.model_ckpt, map_location=map_location))
    model.to(rank)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Define train and test dataset IDs
    train_ids = ['04', '05', '06', '08', '09', '12', '14', '15', '16', '17', '19', '20', '21', '22', '23', '24', '25', '27']
    test_ids = ['01', '11',  '18', '26'] 

    # debug
    # train_ids = ['04', '08']
    # test_ids = ['01']

    train_dataset_list = []
    test_dataset_list = []

    # unzip if not already unzipped
    from scene3d.utils.dataprocess import unzip_hamlyn_data
    unzip_hamlyn_data(args)
    
    # Process train datasets
    for id in train_ids:
        data_root = os.path.join(args.data_dir, f'rectified{id}')
        print(f"Loading training data from {data_root}")
        img_dir = os.path.join(data_root, args.img_tag)
        depth_dir = os.path.join(data_root, args.depth_tag)
        intr_file = os.path.join(args.calibr_dir, id, args.intr_tag)
        intrinsic = read_intrinsics(intr_file)

        dataset = DepthDataset(img_dir, intrinsic, depth_dir)
        train_dataset_list.append(dataset)
    
    # Process test datasets
    for id in test_ids:
        data_root = os.path.join(args.data_dir, f'rectified{id}')
        print(f"Loading test data from {data_root}")
        img_dir = os.path.join(data_root, args.img_tag)
        depth_dir = os.path.join(data_root, args.depth_tag)
        intr_file = os.path.join(args.calibr_dir, id, args.intr_tag)
        intrinsic = read_intrinsics(intr_file)

        dataset = DepthDataset(img_dir, intrinsic, depth_dir)
        test_dataset_list.append(dataset)

    # Choose whether to use separate test datasets or random split
    use_separate_test_datasets = args.separate_test_datasets if hasattr(args, 'separate_test_datasets') else True
    
    use_separate_test_datasets = True #default is false
    if use_separate_test_datasets:
        # Use the separate test datasets
        train_dataset = ConcatDataset(train_dataset_list)
        total_len = len(train_dataset)
        val_len = int(total_len * 0.15)
        train_len = total_len - val_len
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_len, val_len],
            generator=torch.Generator().manual_seed(42))
        print(f"Using train/val datasets with {len(train_dataset)} training samples")
        
        test_dataset = ConcatDataset(test_dataset_list)
        print(f"Using separate test datasets with {len(test_dataset)} samples")
    else:
        # Combine all datasets and do a random split
        dataset_list = train_dataset_list + test_dataset_list
        combined_dataset = ConcatDataset(dataset_list)
        total_len = len(combined_dataset)
        test_len = int(total_len * 0.15)
        train_len = total_len - test_len
        train_dataset, test_dataset = torch.utils.data.random_split(
            combined_dataset, [train_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"Using random split with {test_len} test samples")

    from scene3d.utils.dataprocess import pad_collate_fn
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=pad_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        collate_fn=pad_collate_fn,
        num_workers=4,
        pin_memory=True
    )

    from torch.utils.data import SequentialSampler
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(test_dataset),
        collate_fn=pad_collate_fn,
        num_workers=4,
        pin_memory=True
    )   

    if rank == 0:
        print(f"Training on HAMLYN dataset, total {len(train_dataset)} training samples")

    train_hist = []
    val_hist= [] #unused with update. df stores val hist
    test_hist = []
    loss_history_df = {
        'batch_losses': [],
        'train_losses': [],
        'val_losses': [],
        'test_losses': [],
        'me_losses': [],
        'batch_val_losses': [],
        'batch_test_losses': [],
        'me_val_losses': [],
        'me_test_losses': [],
    }

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        tag=f'lr_{args.lr}_epoch_{epoch+1}'
        
        #init save
        ckpt_dir = os.path.join(args.save_path, f'model_ckpts')
        os.makedirs(ckpt_dir, exist_ok=True)
        date_stamp = datetime.now().strftime("%Y%m%d")            
        save_path = os.path.join(ckpt_dir, f'ckpt_{date_stamp}_epoch{epoch}.pth')
        model_to_save = model.module if world_size > 1 else model
        torch.save(model_to_save.state_dict(), save_path)
        
        loss, test_loss, loss_history_df = train_one_epoch(
            model, args.model_scale, train_loader, val_loader, test_loader, 
            optimizer, criterion, rank, 
            plot=True, epoch=(epoch+1), 
            loss_history_df=loss_history_df,
            tag=tag, save_path=args.save_path
        )
        train_hist.append(loss)
        test_hist.append(test_loss)
        if rank == 0:
            ckpt_dir = os.path.join(args.save_path, f'model_ckpts')
            os.makedirs(ckpt_dir, exist_ok=True)
            date_stamp = datetime.now().strftime("%Y%m%d")            
            
            save_path = os.path.join(ckpt_dir, f'ckpt_{date_stamp}_epoch{epoch+1}.pth')
            model_to_save = model.module if world_size > 1 else model
            torch.save(model_to_save.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")
            
            plt_tag=f'TrainTest_{date_stamp}_e{epoch+1}_lr{args.lr}'
            plot_losses(train_hist, test_hist, loss_history_df, epoch, plt_tag, args.save_path)

    
    if rank == 0:
        # Save loss history to CSV
        loss_history_df_pd = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in loss_history_df.items() ]))
        csv_save_path = os.path.join(args.save_path, f'loss_history_{date_stamp}.csv')
        loss_history_df_pd.to_csv(csv_save_path, index=False)
        print(f"Saved loss history to {csv_save_path}")

    # Optionally Remove data folders
    # for zip_name in rectified_zip_dirs:
    #     dir_id = zip_name.replace('rectified', '').replace('.zip', '')
    #     unzip_path = os.path.join(args.data_dir, f'rectified{dir_id}')
    #     if os.path.exists(unzip_path):
    #         print(f"Removing {unzip_path}")
    #         os.system(f"rm -rf {unzip_path}")

    if world_size > 1:
        cleanup()

def launch_ddp():
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='yvanyin/metric3d')
    parser.add_argument('--model_type', type=str, default='metric3d_vit_large')
    parser.add_argument('--model_scale', type=float, default=0.05) # prescaling from metric3d training
    parser.add_argument('--data_dir', type=str, default='HAMLYN/hamlyn_data')
    parser.add_argument('--separate_test_datasets', type=bool, default=False)
    parser.add_argument('--calibr_dir', type=str, default='HAMLYN/hamlyn_data/calibration')
    parser.add_argument('--img_tag', type=str, default='image01_crop')
    parser.add_argument('--depth_tag', type=str, default='NVFS_crop')
    parser.add_argument('--intr_tag', type=str, default='intrinsics_crop.txt')
    parser.add_argument('--save_path', type=str, default='scene3d/model_training/Metric3D_large_hamlyn')
    parser.add_argument('--model_ckpt', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-7)
    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())
    args = parser.parse_args()
    
    # args.model_ckpt = None #edit here to quickly add ckpt
    
    os.makedirs(args.save_path, exist_ok=True)
    
    if args.gpus == 1:
        print("Running in single GPU mode without spawn.")
        ddp_main(0, 1, args)
    else:
        print(f"Launching DDP on {args.gpus} GPUs")
        mp.spawn(ddp_main, args=(args.gpus, args), nprocs=args.gpus, join=True)
     

if __name__ == '__main__':
    global debug
    debug = False
    launch_ddp()
