import os
import sys
import io
import torch
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from collections import Counter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.tensorboard import SummaryWriter
from diffusion_model.diffusion import DiffusionProcess, Scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusion_model.model_loader import load_sensor_model, load_diffusion
from diffusion_model.skeleton_model import SkeletonTransformer
from diffusion_model.util import (
    prepare_dataset,
    compute_loss,
)


def ensure_dir(path, rank):
    if rank == 0:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    dist.barrier()


def setup(rank, world_size, seed):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12362'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Set the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup():
    dist.destroy_process_group()


def train_sensor_model(rank, args, device, train_loader, val_loader):
    print("Training Sensor model")
    # Set seed for reproducibility within this process
    torch.manual_seed(args.seed + rank)
    sensor_model = load_sensor_model(args, device)
    sensor_model = DDP(sensor_model, device_ids=[rank], find_unused_parameters=True)

    sensor_optimizer = torch.optim.Adam(
        sensor_model.parameters(),
        lr=args.sensor_lr,
        betas=(0.9, 0.98)
    )

    sensor_model_save_dir = os.path.join(args.output_dir, "sensor_model")
    ensure_dir(sensor_model_save_dir, rank)

    sensor_log_dir = os.path.join(sensor_model_save_dir, "sensor_logs")
    ensure_dir(sensor_log_dir, rank)

    if rank == 0:
        writer = SummaryWriter(log_dir=sensor_log_dir)

    best_loss = float('inf')

    for epoch in range(args.sensor_epoch):
        sensor_model.train()
        epoch_train_loss = 0.0

        for _, sensor1, sensor2, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.sensor_epoch} (Training)"):
            sensor1, sensor2, labels = sensor1.to(device), sensor2.to(device), labels.to(device)
            sensor_optimizer.zero_grad()
            output, _ = sensor_model(sensor1, sensor2)

            # 🔍 Add here
            label_indices = labels
            if (label_indices >= output.shape[1]).any() or (label_indices < 0).any():
                print(f"Invalid label index detected!")
                print(f"Labels (argmax): {label_indices}")
                print(f"Model output shape: {output.shape}")
                exit()

            loss = torch.nn.CrossEntropyLoss()(output, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(sensor_model.parameters(), max_norm=1.0)
            sensor_optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)

        # Validation phase
        sensor_model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for _, sensor1, sensor2, labels in tqdm(val_loader,
                                                    desc=f"Epoch {epoch + 1}/{args.sensor_epoch} (Validation)"):
                sensor1, sensor2, labels = sensor1.to(device), sensor2.to(device), labels.to(device)
                output, _ = sensor_model(sensor1, sensor2)

                loss = torch.nn.CrossEntropyLoss()(output, labels)

                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)

        if rank == 0:
            print(
                f"Epoch {epoch + 1}/{args.sensor_epoch}, Average Training Loss: {avg_train_loss}, Average Validation Loss: {avg_val_loss}")
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(sensor_model.state_dict(), os.path.join(sensor_model_save_dir, "best_sensor_model.pth"))
                print(f"Saved best sensor model with Validation Loss: {best_loss}")


def train_skeleton_model(rank, args, device, train_loader, val_loader):
    print("Training Skeleton model")
    # Set seed for reproducibility within this process
    torch.manual_seed(args.seed + rank)
    skeleton_model = SkeletonTransformer(input_size=48, num_classes=2).to(device)
    skeleton_model = DDP(skeleton_model, device_ids=[rank], find_unused_parameters=True)

    skeleton_optimizer = torch.optim.Adam(
        skeleton_model.parameters(),
        lr=args.skeleton_lr,
        betas=(0.9, 0.98)
    )

    scheduler = torch.optim.lr_scheduler.StepLR(skeleton_optimizer, step_size=args.step_size, gamma=0.1)

    skeleton_model_save_dir = os.path.join(args.output_dir, "skeleton_model")
    ensure_dir(skeleton_model_save_dir, rank)

    if rank == 0:
        writer = SummaryWriter(log_dir=skeleton_model_save_dir)

    best_loss = float('inf')
    best_accuracy = 0.0

    for epoch in range(args.skeleton_epochs):
        skeleton_model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for skeleton_data, _, _, labels in tqdm(train_loader,
                                                desc=f"Epoch {epoch + 1}/{args.skeleton_epochs} (Training)"):
            skeleton_data, labels = skeleton_data.to(device), labels.to(device)
            skeleton_optimizer.zero_grad()
            output = skeleton_model(skeleton_data)
            loss = torch.nn.CrossEntropyLoss()(output, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(skeleton_model.parameters(), max_norm=1.0)
            skeleton_optimizer.step()

            epoch_train_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train * 100

        # Validation phase
        skeleton_model.eval()
        epoch_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for skeleton_data, _, _, labels in tqdm(val_loader,
                                                    desc=f"Epoch {epoch + 1}/{args.skeleton_epochs} (Validation)"):
                skeleton_data, labels = skeleton_data.to(device), labels.to(device)
                output = skeleton_model(skeleton_data)
                loss = torch.nn.CrossEntropyLoss()(output, labels)

                epoch_val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(output, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val * 100

        # Adjust the learning rate
        scheduler.step()

        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.skeleton_epochs}, "
                  f"Avg Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% "
                  f"Avg Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
            writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_accuracy = val_accuracy
                torch.save(skeleton_model.state_dict(),
                           os.path.join(skeleton_model_save_dir, "best_skeleton_model.pth"))
                print(
                    f"Saved best skeleton model with Validation Loss: {best_loss:.4f} and Accuracy: {best_accuracy:.2f}%")


def train_diffusion_model(rank, args, device, train_loader, val_loader):
    print("Training Diffusion model")
    torch.manual_seed(args.seed + rank)

    # Load models
    sensor_model = load_sensor_model(args, device)
    diffusion_model = load_diffusion(device)
    skeleton_model = SkeletonTransformer(input_size=48, num_classes=2).to(device)

    # Enable DistributedDataParallel
    sensor_model = DDP(sensor_model, device_ids=[rank], find_unused_parameters=True)
    diffusion_model = DDP(diffusion_model, device_ids=[rank], find_unused_parameters=True)
    skeleton_model = DDP(skeleton_model, device_ids=[rank], find_unused_parameters=True)

    # Optimizers and learning rate schedulers
    diffusion_optimizer = optim.Adam(
        diffusion_model.parameters(),
        lr=args.diffusion_lr,
        eps=1e-8,
        betas=(0.9, 0.98)
    )
    skeleton_optimizer = optim.Adam(skeleton_model.parameters(), lr=args.skeleton_lr, eps=1e-8, betas=(0.9, 0.98))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(diffusion_optimizer, mode='min', factor=0.5, patience=8,
                                                           verbose=False)
    skeleton_scheduler = torch.optim.lr_scheduler.StepLR(skeleton_optimizer, step_size=args.step_size, gamma=0.1)

    # Set up output directories
    diffusion_model_save_dir = os.path.join(args.output_dir, "diffusion_model")
    skeleton_model_save_dir = os.path.join(args.output_dir, "skeleton_model")
    ensure_dir(diffusion_model_save_dir, rank)
    ensure_dir(skeleton_model_save_dir, rank)

    writer = SummaryWriter(log_dir=diffusion_model_save_dir) if rank == 0 else None

    best_diffusion_loss = float('inf')
    best_skeleton_loss = float('inf')
    scaling_factor = 0.01  # Initial scaling factor for skeleton model loss

    # Initialize the diffusion process with the scheduler
    diffusion_process = DiffusionProcess(
        scheduler=Scheduler(sched_type='cosine', T=args.timesteps, step=1, device=device),
        device=device,
        ddim_scale=args.ddim_scale
    )

    for epoch in range(args.epochs):
        diffusion_model.train()
        sensor_model.train()
        epoch_train_loss = 0.0
        epoch_skeleton_loss = 0.0
        correct_train = 0
        total_train = 0

        # Train diffusion model and skeleton model
        for batch_idx, (skeleton, sensor1, sensor2, mask) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} (Training)")):
            skeleton, sensor1, sensor2, mask = skeleton.to(device), sensor1.to(device), sensor2.to(device), mask.to(
                device)
            t = torch.randint(1, args.timesteps, (skeleton.shape[0],), device=device).long()
            output, context = sensor_model(sensor1, sensor2, return_attn_output=True)
            diffusion_optimizer.zero_grad()
            skeleton_optimizer.zero_grad()

            # Compute diffusion model loss and generate x0_pred
            loss, x0_pred = compute_loss(
                args=args,
                model=diffusion_model,
                x0=skeleton,
                context=context,
                label=mask,
                t=t,
                mask=mask,
                device=device,
                diffusion_process=diffusion_process,
                angular_loss=args.angular_loss,
                epoch=epoch,
                rank=rank,
                batch_idx=batch_idx
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
            diffusion_optimizer.step()
            epoch_train_loss += loss.item()

            # Ensure skeleton model is in training mode before calculating skeleton loss
            skeleton_model.train()
            skeleton_output = skeleton_model(x0_pred.detach())
            skeleton_loss = torch.nn.CrossEntropyLoss()(skeleton_output, mask)

            # Apply scaling factor to the skeleton loss
            adjusted_skeleton_loss = scaling_factor * skeleton_loss
            adjusted_skeleton_loss.backward()
            torch.nn.utils.clip_grad_norm_(skeleton_model.parameters(), max_norm=1.0)
            skeleton_optimizer.step()
            epoch_skeleton_loss += adjusted_skeleton_loss.item()

            # Calculate accuracy for skeleton model
            _, predicted = torch.max(skeleton_output, 1)
            total_train += mask.size(0)
            correct_train += (predicted == mask).sum().item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_skeleton_loss = epoch_skeleton_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train

        # Validation phase
        diffusion_model.eval()
        skeleton_model.eval()
        epoch_val_loss = 0.0
        epoch_skeleton_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_idx, (skeleton, sensor1, sensor2, mask) in enumerate(
                    tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} (Validation)")):
                skeleton, sensor1, sensor2, mask = skeleton.to(device), sensor1.to(device), sensor2.to(device), mask.to(
                    device)
                t = torch.randint(1, args.timesteps, (skeleton.shape[0],), device=device).long()
                output, context = sensor_model(sensor1, sensor2, return_attn_output=True)

                # Compute validation loss for diffusion model
                val_loss, x0_pred_val = compute_loss(
                    args=args,
                    model=diffusion_model,
                    x0=skeleton,
                    context=context,
                    label=mask,
                    t=t,
                    mask=mask,
                    device=device,
                    diffusion_process=diffusion_process,
                    angular_loss=args.angular_loss,
                    epoch=epoch,
                    rank=rank,
                    batch_idx=batch_idx
                )
                epoch_val_loss += val_loss.item()

                # Skeleton model validation loss
                skeleton_output_val = skeleton_model(x0_pred_val.detach())
                skeleton_val_loss = torch.nn.CrossEntropyLoss()(skeleton_output_val, mask)
                epoch_skeleton_val_loss += skeleton_val_loss.item()

                _, predicted_val = torch.max(skeleton_output_val, 1)
                total_val += mask.size(0)
                correct_val += (predicted_val == mask).sum().item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_skeleton_val_loss = epoch_skeleton_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        print(f"📊 Skeleton Validation Accuracy: {val_accuracy:.2f}%")
        if avg_val_loss < best_diffusion_loss:
            best_diffusion_loss = avg_val_loss
            if rank == 0:
                torch.save(diffusion_model.state_dict(),
                           os.path.join(diffusion_model_save_dir, "best_diffusion_model.pth"))
                print(f"Saved best diffusion model at {diffusion_model_save_dir}")
            scaling_factor = min(scaling_factor + 0.1, 1.0)

        scheduler.step(avg_val_loss)
        skeleton_scheduler.step(avg_skeleton_val_loss)

        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}, "
                  f"Avg Diffusion Train Loss: {avg_train_loss:.4f}, Avg Diffusion Val Loss: {avg_val_loss:.4f}, "
                  f"Avg Skeleton Train Loss: {avg_skeleton_loss:.4f}, Avg Skeleton Val Loss: {avg_skeleton_val_loss:.4f}, "
                  f"Skeleton Train Accuracy: {train_accuracy:.2f}, Skeleton Val Accuracy: {val_accuracy:.2f}")
            writer.add_scalar('Loss/Diffusion Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Diffusion Validation', avg_val_loss, epoch)
            writer.add_scalar('Loss/Skeleton Train', avg_skeleton_loss, epoch)
            writer.add_scalar('Loss/Skeleton Validation', avg_skeleton_val_loss, epoch)
            writer.add_scalar('Accuracy/Skeleton Train', train_accuracy, epoch)
            writer.add_scalar('Accuracy/Skeleton Validation', val_accuracy, epoch)
            writer.add_scalar('Scaling Factor', scaling_factor, epoch)

            # Save the best models
            if rank == 0 and (epoch + 1) % 300 == 0 or (epoch + 1) == args.epochs:
                diffusion_model_path = os.path.join(diffusion_model_save_dir, f"diffusion_model_epoch_{epoch + 1}.pth")
                skeleton_model_path = os.path.join(skeleton_model_save_dir, f"skeleton_model_epoch_{epoch + 1}.pth")
                torch.save(diffusion_model.state_dict(), diffusion_model_path)
                torch.save(skeleton_model.state_dict(), skeleton_model_path)
                print(f"Saved diffusion model checkpoint at epoch {epoch + 1} to {diffusion_model_path}")
                print(f"Saved skeleton model checkpoint at epoch {epoch + 1} to {skeleton_model_path}")

            if avg_skeleton_val_loss < best_skeleton_loss:
                best_skeleton_loss = avg_skeleton_val_loss
                torch.save(skeleton_model.state_dict(),
                           os.path.join(skeleton_model_save_dir, "best_skeleton_model.pth"))
                print(f"Saved best Skeleton model at {skeleton_model_save_dir}")


def main(rank, args):
    if rank == 0:
        if args.train_sensor_model:
            log_filename = 'sensor_train.log'
        elif args.train_skeleton_model:
            log_filename = 'skeleton_train.log'
        else:
            log_filename = 'diffusion_train.log'

        log_path = os.path.join(args.output_dir, log_filename)
        log_file = open(log_path, 'ab', buffering=0)
        sys.stdout = io.TextIOWrapper(log_file, write_through=True)
        sys.stderr = sys.stdout

    setup(rank, args.world_size, seed=42)
    device = torch.device(f'cuda:{rank}')

    # Prepare the full dataset
    dataset = prepare_dataset(args)
    labels = [dataset[i][3] for i in range(len(dataset))]

    # 🔍 Check number of classes
    print("Checking label format and number of classes...")
    sample_label = labels[0]

    # 🔍 Check number of classes
    print("Checking label format and number of classes...")
    unique_classes = set(labels)
    print(f"Number of classes detected: {len(unique_classes)}")
    print(f"Classes found: {sorted(unique_classes)}")

    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, val_test_idx = next(stratified_split.split(range(len(dataset)), labels))

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_test_idx)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size,
                                                                    rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=args.world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)

    if args.train_skeleton_model:
        train_skeleton_model(rank, args, device, train_loader, val_loader)
    elif args.train_sensor_model:
        train_sensor_model(rank, args, device, train_loader, val_loader)
    else:
        train_diffusion_model(rank, args, device, train_loader, val_loader)

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Training for Diffusion and Sensor Models")
    parser.add_argument('--seed', type=int, default=42, help="seed")

    # Setting up learning rates for the models
    parser.add_argument("--sensor_lr", type=float, default=1e-3, help="Weight decay for sensor regularization")
    parser.add_argument("--skeleton_lr", type=float, default=1e-3, help="Learning rate for training skeleton data")
    parser.add_argument("--diffusion_lr", type=float, default=1e-5, help="Learning rate for training diffusion model")

    # Whether to train the sensor or skeleton model separately
    parser.add_argument("--train_sensor_model", type=eval, choices=[True, False], default=False,
                        help="Set to True to train the sensor model; set to False to train the diffusion model")
    parser.add_argument("--train_skeleton_model", type=eval, choices=[True, False], default=False,
                        help="Set to True to train the skeleton model; Set to False if skeleton model is already trained")

    # Data folders and setting up the parameters for the dataset.py
    parser.add_argument("--overlap", type=int, default=45, help="Overlap for the sliding window dataset")
    parser.add_argument("--window_size", type=int, default=90, help="Window size for the sliding window dataset")
    parser.add_argument("--skeleton_folder", type=str, default="./Own_Data/Labelled_Student_data/Skeleton_Data",
                        help="Path to the skeleton data folder")
    parser.add_argument("--sensor_folder1", type=str,
                        default="./Own_Data/Labelled_Student_data/Accelerometer_Data/W_Accel_Final",
                        help="Path to the first sensor data folder")
    parser.add_argument("--sensor_folder2", type=str,
                        default="./Own_Data/Labelled_Student_data/Accelerometer_Data/P_Accel_Final",
                        help="Path to the second sensor data folder")

    # Epoches to train the models
    parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs to train the diffusion model")
    parser.add_argument("--sensor_epoch", type=int, default=500, help="Number of epochs to train the sensor model")
    parser.add_argument("--skeleton_epochs", type=int, default=200, help="Number of epochs to train the skeleton model")

    parser.add_argument("--sensor_model_path", type=str, default="./models/sensor_model.pth",
                        help="Path to the pre-trained sensor model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")

    parser.add_argument("--step_size", type=int, default=20, help="Step size for weight decay")
    parser.add_argument("--world_size", type=int, default=8, help="Number of GPUs to use for training")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save the trained model")

    # Timesteps to use for diffusion forward or reverse process
    parser.add_argument("--timesteps", type=int, default=10000, help="Number of timesteps for the diffusion process")
    parser.add_argument('--ddim_scale', type=float, default=0.0,
                        help='Scale factor for DDIM (0 for pure DDIM, 1 for pure DDPM)')

    # Whether to use the Angular loss and Lip Reg. modules as proposed.
    parser.add_argument("--angular_loss", type=eval, choices=[True, False], default=False,
                        help="Whether to use angular loss during training")
    parser.add_argument("--lip_reg", type=eval, choices=[True, False], default=True,
                        help="Flag to determine whether to inlcude LR or not")

    parser.add_argument("--predict_noise", type=eval, choices=[True, False], default=True,
                        help="Flag to determine whether to inlcude LR or not")

    args = parser.parse_args()

    mp.spawn(main, args=(args,), nprocs=args.world_size, join=True)
