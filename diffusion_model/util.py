import os
import math
import torch
import imageio
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from torch.nn import functional as F
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from .dataset import SlidingWindowDataset, read_csv_files


def compute_loss(
        args, model, x0, label, context, t, mask=None, noise=None, device="cpu",
        diffusion_process=None, angular_loss=False, lip_reg=False, epoch=None, rank=0, batch_idx=0
):
    """
    Diffusion training loss that supports:
      - predict_noise=True  : model predicts eps (DDPM-style objective)
      - predict_noise=False : model predicts x0 directly (x0 objective)
    """

    assert diffusion_process is not None, "diffusion_process must be provided"

    # Move tensors to device
    x0 = x0.to(device)
    label = label.to(device)          # class label (0/1)
    context = context.to(device)
    t = t.to(device)

    # --- Forward diffusion:
    xt, true_noise = diffusion_process.add_noise(x0, t)   # <-- use returned noise
    xt = xt.to(device)
    true_noise = true_noise.to(device)

    predict_noise = bool(getattr(args, "predict_noise", False))  # <-- main switch

    if predict_noise:
        # Model predicts eps
        pred_noise = model(xt, context, t, sensor_pred=label)
        mse_loss = F.mse_loss(pred_noise, true_noise)

        # Derive x0_pred for visualization / skeleton classifier
        sqrt_a_bar_t = diffusion_process.scheduler.sample_sqrt_a_bar_t(t).to(device)
        sqrt_1_minus_a_bar_t = diffusion_process.scheduler.sample_sqrt_1_minus_a_bar_t(t).to(device)
        x0_pred = (xt - sqrt_1_minus_a_bar_t * pred_noise) / (sqrt_a_bar_t + 1e-8)

    else:
        # Model predicts x0 directly
        x0_pred = model(xt, context, t, sensor_pred=label)
        mse_loss = F.mse_loss(x0_pred, x0)

    total_loss = mse_loss

    # -------------------------
    # Optional: visualization
    # -------------------------
    if epoch is not None and batch_idx == 0 and rank == 0:
        if epoch in [99, 599, 999]:
            sample_idx = 0
            x_original = x0[sample_idx].unsqueeze(0)
            x_gen = x0_pred[sample_idx].unsqueeze(0)

            visualize_skeleton(
                x_original.cpu().detach().numpy(),
                save_path=f'./gif_tl/original_skeleton_animation_{epoch}_sample_{sample_idx}.gif'
            )
            visualize_skeleton(
                x_gen.cpu().detach().numpy(),
                save_path=f'./gif_tl/skeleton_animation_epoch_{epoch}_sample_{sample_idx}.gif'
            )

    # -------------------------
    # Optional angular loss
    # -------------------------
    if angular_loss:
        joint_angles = compute_joint_angles(x0)
        predicted_joint_angles = compute_joint_angles(x0_pred)
        difference = joint_angles - predicted_joint_angles
        angular_loss_value = torch.norm(difference, p='fro')
        total_loss = total_loss + 0.05 * angular_loss_value

    # -------------------------
    # Optional Lipschitz reg
    # (perturb context only, keep xt the same)
    # -------------------------
    if lip_reg:
        noisy_context = add_random_noise(context.clone(), noise_std=0.01, noise_fraction=0.2)

        if predict_noise:
            pred_noise_lr = model(xt, noisy_context, t, sensor_pred=label)
            lip_reg_loss = F.mse_loss(pred_noise_lr, true_noise)
        else:
            x0_pred_lr = model(xt, noisy_context, t, sensor_pred=label)
            lip_reg_loss = F.mse_loss(x0_pred_lr, x0_pred.detach())

        total_loss = total_loss + 0.05 * lip_reg_loss

    return total_loss, x0_pred



def add_random_noise(context, noise_std=0.01, noise_fraction=0.2):
    num_samples = context.size(0)
    num_noisy_samples = int(noise_fraction * num_samples)
    noisy_indices = torch.randperm(num_samples)[:num_noisy_samples]

    noise = torch.randn_like(context[noisy_indices]) * noise_std

    context[noisy_indices] += noise

    return context


def frobenius_norm_loss(predicted, target):
    # Calculate the Frobenius norm loss
    return torch.norm(predicted - target, p='fro')


def min_max_scale(data, data_min, data_max, feature_range=(0, 1)):
    data_min = np.array(data_min)
    data_max = np.array(data_max)

    scale = (feature_range[1] - feature_range[0]) / (data_max - data_min + 1e-8)
    min_range = feature_range[0]

    return scale * (data - data_min) + min_range


def prepare_dataset(args):
    skeleton_folder = args.skeleton_folder
    sensor_folder1 = args.sensor_folder1
    sensor_folder2 = args.sensor_folder2

    skeleton_data = read_csv_files(skeleton_folder)
    sensor_data1 = read_csv_files(sensor_folder1)
    sensor_data2 = read_csv_files(sensor_folder2)

    # Find common files across all three directories
    common_files = list(set(skeleton_data.keys()).intersection(set(sensor_data1.keys()), set(sensor_data2.keys())))

    if not common_files:
        raise ValueError("No common files found across the skeleton, sensor1, and sensor2 directories.")

    # Ensure consistent column sizes (96 columns for skeleton data)
    for file in common_files:
        if skeleton_data[file].shape[1] == 97:
            skeleton_data[file] = skeleton_data[file].iloc[:, 1:]  # Drop the first column

    # Extract activity codes from file names
    activity_codes = sorted(set(file.split('A')[1][:2].lstrip('0') for file in common_files))

    label_encoder = OneHotEncoder(sparse_output=False)
    label_encoder.fit([[code] for code in activity_codes])

    # **Add this line here**
    n_classes = len(label_encoder.categories_[0])
    print(f"Activity codes: {activity_codes}, Number of classes: {n_classes}")

    window_size = args.window_size
    overlap = args.overlap

    # Instantiate the dataset with Min-Max scaling applied at each window level
    dataset = SlidingWindowDataset(
        skeleton_data=skeleton_data,
        sensor1_data=sensor_data1,
        sensor2_data=sensor_data2,
        common_files=common_files,
        window_size=window_size,
        overlap=overlap,
        label_encoder=label_encoder,
        scaling="minmax"  # Use Min-Max scaling for each window segment
    )

    return dataset


def sample_by_t(tensor_to_sample, timesteps, x_shape):
    batch_size = timesteps.shape[0]
    timesteps = timesteps.to(tensor_to_sample.device)
    sampled_tensor = tensor_to_sample.gather(0, timesteps)
    sampled_tensor = torch.reshape(sampled_tensor, (batch_size,) + (1,) * (len(x_shape) - 1))
    return sampled_tensor


def extract_joint_subset(positions):
    """
    Extract only the subset of key points for computing joint angles based on the joint index mapping.
    The input tensor should remain [batch_size, 90, 96] but only specific columns will be used for angle computation.

    Args:
        positions (torch.Tensor): The full skeleton tensor of shape [batch_size, 90, 96].

    Returns:
        torch.Tensor: Subset tensor containing only the columns corresponding to relevant joints.
    """
    joint_indices = {
        5: [15, 16, 17],  # Left shoulder
        6: [18, 19, 20],  # Left elbow
        7: [21, 22, 23],  # Left wrist
        12: [36, 37, 38],  # Right shoulder
        13: [39, 40, 41],  # Right elbow
        14: [42, 43, 44],  # Right wrist
        18: [54, 55, 56],  # Left hip
        19: [57, 58, 59],  # Left knee
        20: [60, 61, 62],  # Left ankle
        22: [66, 67, 68],  # Right hip
        23: [69, 70, 71],  # Right knee
        24: [72, 73, 74],  # Right ankle
    }

    selected_columns = sum(joint_indices.values(), [])

    return positions[:, :, selected_columns]


def compute_joint_angles(positions):
    # Indices of the joints of interest for computing angles

    joint_pairs = torch.tensor([
        [3, 4, 5],  # Left shoulder, elbow, wrist
        [6, 7, 8],  # Right shoulder, elbow, wrist
        [9, 10, 11],  # Left hip, knee, ankle
        [12, 13, 14]  # Right hip, knee, ankle
    ], device=positions.device)

    batch_size, num_frames, _ = positions.shape

    # Account for potential padding: remove extra columns
    # Calculate how many joints we have based on columns
    num_joints = (_ // 3)

    # If extra padding columns exist, slice them off
    positions = positions[:, :, :num_joints * 3]

    # Reshape positions to (batch_size, num_frames, num_joints, 3)
    positions = positions.view(batch_size, num_frames, num_joints, 3)

    # Process in smaller chunks if the tensor is large
    chunk_size = 100  # Adjust based on memory capacity
    angles = []

    for i in range(0, num_frames, chunk_size):
        positions_chunk = positions[:, i:i + chunk_size]

        # Compute vectors for the joint pairs
        vectors1 = positions_chunk[:, :, joint_pairs[:, 1]] - positions_chunk[:, :, joint_pairs[:, 0]]
        vectors2 = positions_chunk[:, :, joint_pairs[:, 1]] - positions_chunk[:, :, joint_pairs[:, 2]]

        # Compute dot product
        dot_product = torch.sum(vectors1 * vectors2, dim=-1)

        # Compute norms
        norm1 = torch.norm(vectors1, dim=-1)
        norm2 = torch.norm(vectors2, dim=-1)

        # Avoid division by zero
        denominator = norm1 * norm2
        valid_denominator = denominator != 0

        # Compute cosine of angles where denominator is valid
        cosine_angles = torch.zeros_like(dot_product)
        epsilon = 1e-6
        denominator = torch.clamp(denominator, min=epsilon)
        cosine_angles[valid_denominator] = dot_product[valid_denominator] / denominator[valid_denominator]

        # Clamp values to avoid numerical instability
        cosine_angles = torch.clamp(cosine_angles, -1.0 + 1e-7, 1.0 - 1e-7)

        # Compute angles in radians
        chunk_angles = torch.acos(cosine_angles)
        chunk_angles[~valid_denominator] = 0

        angles.append(chunk_angles)

    # Concatenate all angle chunks
    return torch.cat(angles, dim=1)


def linear_noise_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return np.linspace(beta_start, beta_end, timesteps)


def cosine_noise_schedule(timesteps, s=0.008):
    steps = np.arange(timesteps + 1) / timesteps
    alphas_cumprod = np.cos((steps + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas


def quadratic_noise_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return np.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_noise_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = np.linspace(-6, 6, timesteps)
    return beta_start + (beta_end - beta_start) / (1 + np.exp(-betas))


def get_noise_schedule(schedule_type, timesteps, beta_start=0.0001, beta_end=0.02):
    if schedule_type == 'linear':
        return linear_noise_schedule(timesteps, beta_start, beta_end)
    elif schedule_type == 'cosine':
        return cosine_noise_schedule(timesteps)
    elif schedule_type == 'quadratic':
        return quadratic_noise_schedule(timesteps, beta_start, beta_end)
    elif schedule_type == 'sigmoid':
        return sigmoid_noise_schedule(timesteps, beta_start, beta_end)
    else:
        raise ValueError(f"Unknown noise schedule type: {schedule_type}")


def create_stratified_split(dataset, test_size=0.3, val_size=0.5, random_state=42):
    # Extract labels for stratified splitting
    labels = [label.argmax().item() for _, _, _, label in dataset]  # Assumes labels are one-hot encoded

    # Perform the initial stratified split to get train and (validation + test) indices
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_test_idx = next(stratified_split.split(np.zeros(len(labels)), labels))

    # Perform another stratified split on the (validation + test) indices
    stratified_val_test_split = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    val_idx, test_idx = next(
        stratified_val_test_split.split(np.zeros(len(val_test_idx)), [labels[i] for i in val_test_idx]))

    # Map back to the original indices
    val_idx = [val_test_idx[i] for i in val_idx]
    test_idx = [val_test_idx[i] for i in test_idx]

    return train_idx, val_idx, test_idx


def calculate_fid(real_activations, generated_activations):
    real_activations = np.concatenate(real_activations, axis=0)
    generated_activations = np.concatenate(generated_activations, axis=0)

    real_activations = real_activations.reshape(real_activations.shape[0], -1)
    generated_activations = generated_activations.reshape(generated_activations.shape[0], -1)

    mu_real = np.mean(real_activations, axis=0)
    sigma_real = np.cov(real_activations, rowvar=False)
    mu_generated = np.mean(generated_activations, axis=0)
    sigma_generated = np.cov(generated_activations, rowvar=False)

    diff = mu_real - mu_generated
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_generated, disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm(sigma_real @ sigma_generated + np.eye(sigma_real.shape[0]) * 1e-6)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_generated) - 2 * np.trace(covmean)
    return fid


def get_time_embedding(timestep, dtype):
    half_dim = 320 // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim) * -emb)
    emb = timestep * emb
    emb = np.concatenate((np.sin(emb), np.cos(emb)))
    return torch.tensor(emb, dtype=dtype)


def get_file_path(file_name):
    return os.path.join(os.path.dirname(__file__), file_name)


def rescale(tensor, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    tensor = ((tensor - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    if clamp:
        tensor = torch.clamp(tensor, new_min, new_max)
    return tensor


def visualize_skeleton(positions, save_path='skeleton_animation.gif'):
    """
    Visualize the skeleton animation based on the adjusted keypoints.
    The new joint order is:
    Head -> Neck -> Left Arm -> Right Arm -> Spine -> Left Leg -> Right Leg
    """

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

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

    # 💡 Add this line before saving the gif
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the frames as a GIF with a duration of 0.2 seconds per frame (5 fps)
    imageio.mimsave(save_path, frames, duration=0.2)  # Adjust duration for frame speed
    print(f'GIF saved as {save_path}')
