# MobileManipVLA
Mobile Manipulation Benchmark based on IsaacSim and IsaacLab

## Folder
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
            └── GenieSimAssets
            └── Isaac
        └── partnet
            └── dataset
            └── process
        └── unidoor
            └── dataset
            └── process
        └── ycb
        └── xhand_robot
        └── g1_robot
    |
    └── Logs
        └── MobileManiDataset
            └── XHand_Robot
                └── Best_0
                    └── Open
                        └── partnet
                        └── unidoor
                        └── ycb
                    └── Close
            └── G1_Robot
```

## Installation

### 1. Create virtual environment
```bash
conda create -n mobilemanibench python==3.10 -y
conda activate mobilemanibench
```

### 2. Update GLIBC (Ubuntu20.04)
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