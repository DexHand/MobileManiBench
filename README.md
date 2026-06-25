# <p align="center"> MobileManiBench: Simplifying Model Verification <br> for Mobile Manipulation </p>

### <p align="center"> Microsoft Research Asia </p>

### <p align="center">[ArXiv](https://arxiv.org/abs/2602.05233) | [Website](https://dexhand.github.io/MobileManiBench_Website/)

<p align="center">
  <img width="100%" src="teaser.png"/>
</p>
Overview of <b>MobileManiBench</b>. It features 2 mobile-based robots: the G1 robot with a parallel gripper and the XHand robot with a dexterous hand. 
The benchmark includes 640 articulated and holistic objects across 20 categories and supports 5 mobile manipulation skills—open, close, pull, push, and pick—enabling over 100 tasks. 
To efficiently scale data generation while ensuring task success, we train a universal <b>MobileManiRL</b> policy for each robot-object-skill triplet and generate <b>MobileManiDataset</b> across 100 realistic scenes with 300K trajectories and 3 data modalities—language instructions, multi-view RGB-depth-segmentation images, synchronized object/robot states and actions. 
MobileManiBench offers a flexible testbed to accelerate model innovation and data-efficiency research for VLA models.
</p>


# I. TODO List
- [x] MobileManiBench code and assets for IssacSim.
- [x] MobileManiRL checkpoints.
- [x] MobileManiDataset.
- [x] MobileManiVLA.


# II. Get Started
## Folder Structure:
```
PROJECT
    └── MobileManiBench
        └── MobileManiVLA
        └── source
            └── isaaclab_tasks/isaaclab_tasks/direct
                └── xhand_robot
                └── g1_robot
        └── unimanip(MobileManiRL)
            └── configs
                └── data
                └── train
            └── rsl_ppo
                └── train.py
                └── play.py
            └── utils
                └── env_model.py
                └── room_model.py
                └── dataset_model.py
                └── partnet_model.py
                └── unidoor_model.py
                └── ycb_model.py
    |
    └── Assets
        └── room
            └── GenieSim
            └── IsaacSim
        └── partnet
            └── dataset
            └── process
        └── unidoor
            └── dataset
            └── process
        └── ycb
        └── xhand_robot_rotate
        └── g1_robot_rotate
    |
    └── Logs
        └── MobileManiDataset
            └── G1_Robot
                └── Open
                    └── partnet
                        └── box
                        └── ...
                    └── unidoor
                        └── cabinet
                        └── ...
                    └── ycb
                        └── ycb
                └── Close
            └── XHand_Robot
```

## Installation
Recommend: Driver 550.x; Cuda 12.x; Ubuntu 22.04 LTS

### 1. Create virtual environment
```bash
conda create -n mobilemanibench python==3.10 -y
conda activate mobilemanibench
```

### 2. Update GLIBC (For Ubuntu20.04)
Modify apt sources:
```bash
sudo nano /etc/apt/sources.list
```
Add required mirror:
```
deb http://mirrors.aliyun.com/ubuntu/ jammy main
```
Update and install libc6:
```bash
sudo apt-get update
sudo apt install libc6
```

### 3. Install isaacsim
```bash
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```

### 4. Install isaaclab and unimanip
```bash
git clone https://github.com/DexHand/MobileManiBench.git
cd MobileManiBench
./isaaclab.sh --install
pip install -e .
```

### 5. Install MobileManiVLA
```bash
cd MobileManiVLA
pip install -e .
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
```


### Download Assets

Download USD files for robots, objects, and scenes from [Hugging Face](https://huggingface.co/datasets/arnoldland/MobileManiBench).
```
cd PROJECT
hf download arnoldland/MobileManiBench Assets.zip --repo-type dataset --local-dir ./
unzip Assets.zip
```

# III. Train MobileManiRL and Generate MobileManiDataset
## Step1: Train&Test Dedicated MobileManiRL Policy:
```
cd PROJECT/MobileManiBench/unimanip/rsl_ppo/
```

Train 0-9 objects in 10 runs, using G1/XHand robots, 
with (open, pull, pick)/(close, push) skills, on ycb/partnet/unidoor objects, on local devices:
```
bash train_parallel.sh 0 9 10 Isaac-G1-Robot-Direct-v0 train_g1_robot_open_best_0.yaml ycb ycb local
bash train_parallel.sh 0 9 10 Isaac-XHand-Robot-Direct-v0 train_xhand_robot_open_best_0.yaml partnet laptop local

bash train_parallel.sh 0 9 10 Isaac-G1-Robot-Direct-v0 train_g1_robot_close_best_0.yaml partnet cart local
bash train_parallel.sh 0 9 10 Isaac-XHand-Robot-Direct-v0 train_xhand_robot_close_best_0.yaml unidoor cabinet local
```

## Step2: Generate MobileManiDataset:
```
cd PROJECT/MobileManiBench/unimanip/rsl_ppo/
```

Record 0-9 objects in 10 runs, across 0-15 training scenes, using G1/XHand robots, 
with (open, pull, pick)/(close, push) skills, on ycb/partnet/unidoor objects, from ppo policies:
```
bash record_parallel.sh 0 9 10 0 15 Isaac-G1-Robot-Direct-v0 train_g1_robot_open_best_0.yaml ycb ycb ppo
bash record_parallel.sh 0 9 10 0 15 Isaac-XHand-Robot-Direct-v0 train_xhand_robot_open_best_0.yaml partnet laptop ppo

bash record_parallel.sh 0 9 10 0 15 Isaac-G1-Robot-Direct-v0 train_g1_robot_close_best_0.yaml partnet cart ppo
bash record_parallel.sh 0 9 10 0 15 Isaac-XHand-Robot-Direct-v0 train_xhand_robot_close_best_0.yaml unidoor cabinet ppo
```


# IV. Download MobileManiDataset

Download the entire MobileManiDataset from [Hugging Face](https://huggingface.co/datasets/arnoldland/MobileManiBench).
```
cd PROJECT/Logs
hf download arnoldland/MobileManiBench --repo-type dataset --local-dir ./
```

Download one robot/task/object.tar from [Hugging Face](https://huggingface.co/datasets/arnoldland/MobileManiBench).
```
cd PROJECT/Logs
hf download arnoldland/MobileManiBench MobileManiDataset/G1_Robot/Open/partnet/box.tar --repo-type dataset --local-dir ./
tar -xf MobileManiDataset/G1_Robot/Open/partnet/box.tar -C MobileManiDataset/
```


# V. Test Pre-Trained MobileManiVLA (Coming Soon)
Check record_infer_vla.sh
```
cd PROJECT/MobileManiBench/unimanip/rsl_ppo/
bash record_infer_vla.sh
```

# VI. Train MobileManiVLA from scratch (Coming Soon)