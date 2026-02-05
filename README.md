# sensor-skeleton-diff

## How to Run

### 1) Create the data folders

From the project root, create this structure:

```text
Own_Data/
└── Labelled_Student_data/
    ├── Accelerometer_Data/
    │   ├── W_Accel_Final/
    │   └── P_Accel_Final/
    └── Skeleton_Data/
```

Put your files here:
- `./Own_Data/Labelled_Student_data/Accelerometer_Data/W_Accel_Final/`
- `./Own_Data/Labelled_Student_data/Accelerometer_Data/P_Accel_Final/`
- `./Own_Data/Labelled_Student_data/Skeleton_Data/`

---

### 2) Train sensor model

```bash
nohup python3 -u train.py \
  --train_sensor_model=True \
  --train_skeleton_model=False \
  --world_size=1 \
  --output_dir=./results \
  > sensor_train.log 2>&1 &
```

---

### 3) Train skeleton model

```bash
nohup python3 -u train.py \
  --train_sensor_model=False \
  --train_skeleton_model=True \
  --world_size=1 \
  --output_dir=./results \
  > skeleton_train.log 2>&1 &
```

---

### 4) Train diffusion model

```bash
nohup python3 -u train.py \
  --train_sensor_model=False \
  --train_skeleton_model=False \
  --world_size=1 \
  --output_dir=./results \
  > diffusion_train.log 2>&1 &
```

---

### 5) Generate samples

```bash
python3 generate.py \
  --sensor_folder1 ./Own_Data/Labelled_Student_data/Accelerometer_Data/W_Accel_Final \
  --sensor_folder2 ./Own_Data/Labelled_Student_data/Accelerometer_Data/P_Accel_Final \
  --skeleton_model_path ./results/skeleton_model/best_skeleton_model.pth \
  --output_dir ./results \
  --test_diffusion_model=True \
  --ddim_scale=0.0 \
  --batch_size=1 \
  --timesteps=1000
```
