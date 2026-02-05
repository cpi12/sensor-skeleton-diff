import os
import torch
import argparse
import imageio
import random
import numpy as np
from diffusion_model import Diffusion1D
from diffusion_model.model_loader import load_sensor_model, load_diffusion_model_for_testing
from diffusion_model.skeleton_model import SkeletonTransformer
from diffusion_model.diffusion import DiffusionProcess, Scheduler
from torch.utils.data import DataLoader
from diffusion_model.util import prepare_dataset
import matplotlib.pyplot as plt


def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def visualize_skeleton(positions, save_path='skeleton_animation.gif'):
    # Adjusted connections based on the new key joint order
    connections = [
        # Head and Neck
        (0, 1),  # Head -> Neck

        # Left Arm
        (1, 2),  # Neck -> Left Shoulder
        (2, 3),  # Left Shoulder -> Left Elbow
        (3, 4),  # Left Elbow -> Left Wrist

        # Right Arm
        (1, 5),  # Neck -> Right Shoulder
        (5, 6),  # Right Shoulder -> Right Elbow
        (6, 7),  # Right Elbow -> Right Wrist

        # Spine
        (1, 8),  # Neck -> Spine Chest
        (8, 9),  # Spine Chest -> Pelvis

        # Left Leg
        (9, 10),  # Pelvis -> Left Hip
        (10, 11),  # Left Hip -> Left Knee
        (11, 12),  # Left Knee -> Left Ankle

        # Right Leg
        (9, 13),  # Pelvis -> Right Hip
        (13, 14),  # Right Hip -> Right Knee
        (14, 15)  # Right Knee -> Right Ankle
    ]

    frames = []
    sample_idx = 0

    # Loop through all 90 frames
    for frame_idx in range(90):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Remove background and axes
        ax.set_facecolor('white')
        ax.grid(False)
        ax.set_axis_off()

        for joint1, joint2 in connections:
            joint1_coords = positions[sample_idx, frame_idx, joint1 * 3:(joint1 * 3) + 3]
            joint2_coords = positions[sample_idx, frame_idx, joint2 * 3:(joint2 * 3) + 3]

            if len(joint1_coords) < 3 or len(joint2_coords) < 3:
                continue

            xs = [joint1_coords[0], joint2_coords[0]]
            ys = [joint1_coords[1], joint2_coords[1]]
            zs = [joint1_coords[2], joint2_coords[2]]

            # Plot the bones as dark blue lines
            ax.plot(xs, ys, zs, marker='o', color='darkblue')

            # Plot the joints as red dots
            ax.scatter(joint1_coords[0], joint1_coords[1], joint1_coords[2], color='red', s=50)  # Joint 1
            ax.scatter(joint2_coords[0], joint2_coords[1], joint2_coords[2], color='red', s=50)  # Joint 2

        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])

        # Set a rotated view angle for better depth perception
        ax.view_init(elev=-90, azim=-90)  # Adjust azimuth and elevation for better 3D perception

        # Capture the frame
        plt.tight_layout()
        fig.canvas.draw()

        # Convert to a numpy array and add to frames list
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

        # Close the figure to save memory
        plt.close(fig)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the frames as a GIF with a duration of 0.2 seconds per frame (5 fps)
    imageio.mimsave(save_path, frames, duration=0.2)
    print(f'GIF saved as {save_path}')


def check_for_nans(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaNs detected in {name}")
    else:
        print(f"No NaNs in {name}")

def generate_samples(args, sensor_model, diffusion_model, device):
    dataset = prepare_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    diffusion_process = DiffusionProcess(
        scheduler=Scheduler(sched_type='cosine', T=args.timesteps, step=1, device=device),
        device=device,
        ddim_scale=args.ddim_scale
    )

    generated_samples = []
    sensor_model.eval()
    diffusion_model.eval()

    with torch.no_grad():
        _, sensor1, sensor2, label = next(iter(dataloader))

        if label.ndim == 2:
            label_index = torch.argmax(label, dim=1)
        else:
            label_index = label.long()

        sensor1, sensor2 = sensor1.to(device), sensor2.to(device)
        label_index = label_index.to(device)

        _, context = sensor_model(sensor1, sensor2, return_attn_output=True)
        check_for_nans(context, "context")

        context = context.to(device)

        generated_sample = diffusion_process.generate(
            model=diffusion_model,
            context=context,
            label=label_index,
            shape=(args.batch_size, 90, 48),
            steps=args.timesteps,
            predict_noise=True
        )

        check_for_nans(generated_sample, "generated_sample")
        generated_samples.append(generated_sample.cpu())

    generated_samples = torch.cat(generated_samples, dim=0)
    return generated_samples



def predict_classes(generated_samples, skeleton_model, device):
    skeleton_model.eval()

    with torch.no_grad():
        # Assuming generated_samples have been normalized to [0, 1] already
        generated_samples = generated_samples.to(device)
        outputs = skeleton_model(generated_samples)
        _, predicted_classes = torch.max(outputs, 1)

    return predicted_classes.cpu()


def load_skeleton_model(skeleton_model_path, skeleton_model):
    # Load the checkpoint
    checkpoint = torch.load(skeleton_model_path, map_location="cpu")

    # Remove the "module." prefix from keys if it exists (due to DistributedDataParallel)
    new_state_dict = {}
    for key in checkpoint:
        new_key = key.replace("module.", "")  # Remove the "module." prefix
        new_state_dict[new_key] = checkpoint[key]

    # Load the modified state dict
    skeleton_model.load_state_dict(new_state_dict, strict=False)
    return skeleton_model


def main(args):
    # Set the seed for reproducibility
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained models
    sensor_model = load_sensor_model(args, device)

    diffusion_model = load_diffusion_model_for_testing(device, args.output_dir, args.test_diffusion_model)

    skeleton_model = SkeletonTransformer(input_size=48, num_classes=2).to(device)
    skeleton_model = load_skeleton_model(args.skeleton_model_path, skeleton_model)

    # Generate samples
    print("Generating samples based on sensor inputs...")
    generated_samples = generate_samples(args, sensor_model, diffusion_model, device)

    visualize_skeleton(
        generated_samples.cpu().detach().numpy(),
        save_path=f'./gif_tl/generatecode_generated_skeleton_animation_{1}.gif')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and Classify Skeleton Samples based on Sensor Input")

    # Add necessary arguments
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--skeleton_model_path", type=str, default="./results/skeleton_model/best_skeleton_model.pth",
                        help="Path to the trained skeleton model")
    parser.add_argument("--sensor_folder1", type=str,
                        default="./Own_Data/Labelled_Student_data/Accelerometer_Data/Meta_wrist",
                        help="Path to the first sensor data folder")
    parser.add_argument("--sensor_folder2", type=str,
                        default="./Own_Data/Labelled_Student_data/Accelerometer_Data/Meta_hip",
                        help="Path to the second sensor data folder")
    parser.add_argument("--window_size", type=int, default=90, help="Window size for the sliding window dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generating samples")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of timesteps for the diffusion process")
    parser.add_argument("--train_sensor_model", type=eval, choices=[True, False], default=False,
                        help="Set to True to train the sensor model; set to False as this is inference")
    parser.add_argument('--ddim_scale', type=float, default=1.0,
                        help='Scale factor for DDIM (0 for pure DDIM, 1 for pure DDPM)')
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save the trained model")
    parser.add_argument("--dataset_type", type=str, default="Own_data", help="Dataset type")
    parser.add_argument("--skeleton_folder", type=str, default="./Own_Data/Labelled_Student_data/Skeleton_Data",
                        help="Path to the skeleton data folder")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap for the sliding window dataset")
    parser.add_argument("--test_diffusion_model", type=eval, choices=[True, False], default=True,
                        help="Whether to test the diffusion mode or not")

    args = parser.parse_args()

    main(args)