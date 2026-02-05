import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from collections import defaultdict

def read_csv_files(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    data = {}
    for file in files:
        file_path = os.path.join(folder, file)
        try:
            data[file] = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            print(f"Skipped empty or unreadable file: {file}")
        except Exception as e:
            print(f"Skipped file {file} due to error: {e}")
    return data

def handle_nan_and_scale(data, scaling_method="standard"):
    if np.all(np.isnan(data), axis=0).any():
        data[:, np.all(np.isnan(data), axis=0)] = 0

    col_mean = np.nanmean(data, axis=0)
    nan_mask = np.isnan(data)
    data[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    return data

def to_one_hot(label, num_classes):
    one_hot = np.zeros(num_classes)
    one_hot[label] = 1
    return one_hot

def adjust_keypoints(skeleton_window, key_joint_indexes, joint_order):
    adjusted_skeleton = []
    for joint_index in joint_order:
        if joint_index in key_joint_indexes:
            start_idx = key_joint_indexes.index(joint_index) * 3
            adjusted_skeleton.append(skeleton_window[:, start_idx:start_idx + 3])
    return np.hstack(adjusted_skeleton)

class SlidingWindowDataset(Dataset):
    def __init__(self, skeleton_data, sensor1_data, sensor2_data, common_files, window_size, overlap, label_encoder, scaling="minmax"):
        self.skeleton_data = skeleton_data
        self.sensor1_data = sensor1_data
        self.sensor2_data = sensor2_data
        self.common_files = list(common_files)
        self.window_size = window_size
        self.overlap = overlap
        self.label_encoder = label_encoder
        self.scaling = scaling
        self.key_joint_indexes = [0, 2, 3, 5, 6, 7, 12, 13, 14, 18, 19, 20, 22, 23, 24, 26]
        self.joint_order = [26, 3, 5, 6, 7, 12, 13, 14, 2, 0, 18, 19, 20, 22, 23, 24]
        self.skeleton_windows, self.sensor1_windows, self.sensor2_windows, self.labels = self._create_windows()

    def _normalize_to_tensor(self, skeleton_window, ref_joint_1=3, ref_joint_2=6, target_length=1.0):
        ref_bone_lengths = np.linalg.norm(
            skeleton_window[:, ref_joint_1 * 3:ref_joint_1 * 3 + 3] - skeleton_window[:, ref_joint_2 * 3:ref_joint_2 * 3 + 3],
            axis=1
        )
        ref_bone_lengths[ref_bone_lengths == 0] = 1e-6
        scale_factors = target_length / ref_bone_lengths[:, np.newaxis]
        skeleton_window_normalized = skeleton_window * scale_factors
        return torch.tensor(skeleton_window_normalized, dtype=torch.float32)

    def _create_windows(self):
        skeleton_windows, sensor1_windows, sensor2_windows, labels = [], [], [], []
        step = self.window_size - self.overlap

        for file in self.common_files:
            skeleton_df = self.skeleton_data[file]
            sensor1_df = self.sensor1_data[file]
            sensor2_df = self.sensor2_data[file]
            activity_code = file.split('A')[1][:2].lstrip('0')
            fall_activities = {"10", "11", "12", "13", "14"}
            label = 1 if activity_code in fall_activities else 0
            num_classes = 2
            num_windows = (len(skeleton_df) - self.window_size) // step + 1

            for i in range(num_windows):
                start = i * step
                end = start + self.window_size
                if end > len(skeleton_df):
                    continue
                skeleton_window = skeleton_df.iloc[start:end, :].values
                sensor1_window = sensor1_df.iloc[start:end, -3:].values
                sensor2_window = sensor2_df.iloc[start:end, -3:].values

                if skeleton_window.shape[1] == 97:
                    skeleton_window = skeleton_window[:, 1:]

                joint_indices = np.array(self.key_joint_indexes)
                final_indices = np.concatenate([[i * 3, i * 3 + 1, i * 3 + 2] for i in joint_indices])
                skeleton_window = skeleton_window[:, final_indices]

                if skeleton_window.shape[0] != self.window_size or sensor1_window.shape[0] != self.window_size or sensor2_window.shape[0] != self.window_size:
                    continue

                skeleton_window = adjust_keypoints(skeleton_window, self.key_joint_indexes, self.joint_order)
                skeleton_window = handle_nan_and_scale(skeleton_window, scaling_method=self.scaling)
                sensor1_window = handle_nan_and_scale(sensor1_window, scaling_method=self.scaling)
                sensor2_window = handle_nan_and_scale(sensor2_window, scaling_method=self.scaling)

                skeleton_window = self._normalize_to_tensor(skeleton_window)

                skeleton_windows.append(skeleton_window)
                sensor1_windows.append(sensor1_window)
                sensor2_windows.append(sensor2_window)
                labels.append(label)

        # Oversampling
        class_indices = defaultdict(list)
        for idx, lbl in enumerate(labels):
            lbl_index = lbl
            class_indices[lbl_index].append(idx)

        skeleton_windows_oversampled, sensor1_windows_oversampled, sensor2_windows_oversampled, labels_oversampled = [], [], [], []

        for lbl, indices in class_indices.items():
            if len(indices) < 2000:
                additional_indices = random.choices(indices, k=1500 - len(indices))
                selected_indices = indices + additional_indices
            else:
                selected_indices = indices

            for idx in selected_indices:
                skeleton_windows_oversampled.append(skeleton_windows[idx])
                sensor1_windows_oversampled.append(sensor1_windows[idx])
                sensor2_windows_oversampled.append(sensor2_windows[idx])
                labels_oversampled.append(lbl)

        return skeleton_windows_oversampled, sensor1_windows_oversampled, sensor2_windows_oversampled, labels_oversampled

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        skeleton_window = self.skeleton_windows[idx]
        sensor1_window = torch.tensor(self.sensor1_windows[idx], dtype=torch.float32)
        sensor2_window = torch.tensor(self.sensor2_windows[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return (
            skeleton_window,
            sensor1_window,
            sensor2_window,
            label,
        )
