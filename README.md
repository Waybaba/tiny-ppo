
# **Environment Setup**

Before you begin, please ensure that the following environment variables are correctly set: `UDATADIR`, `UPRJDIR`, and `UOUTDIR`. The operations you perform will only modify these three directories on your device.

Here's an example setup:

```bash
# Example directory paths
export UDATADIR=~/data # directory for dataset
export UPRJDIR=~/code # directory for code
export UOUTDIR=~/output # directory for outputs such as logs

# Example API Key and Worker settings
export WANDB_API_KEY="xxx360492802218be41f76xxxx" # your Weights & Biases API key
export NUM_WORKERS=0 # number of workers to use

# Create directories if they do not exist
mkdir -p $UDATADIR $UPRJDIR $UOUTDIR

```

**Note:** **[Weights & Biases (wandb)](https://docs.wandb.ai/quickstart)** is a fantastic tool for visualization. It serves a similar purpose to TensorBoard, but offers additional functionality. Please obtain your API key from your Weights & Biases account to make use of these features.

## **Python Environment Setup**

We provide three methods to set up the Python environment for this project:

- Using the Development Environment in VS Code (Most Recommended)
- Using Docker (Recommended)
- Using Pip or Conda

### **Using the Development Environment in VS Code (Most Recommended)**

If you're a Visual Studio Code (VS Code) user, we highly recommend this method for its simplicity. You can set up all the environment requirements for this project in just one step.

1. Open this project in VS Code.
2. Install the "Dev Containers" extension.
3. Press `Cmd/Ctrl+Shift+P` to open the command palette, then select `Dev Container: Rebuild and Reopen in Container`.

Note: The configuration for Dev Containers is stored in the `.devcontainer` folder, which is included in our project. You can modify this if you need to add or remove certain libraries.

Further details and instructions about Dev Containers in VS Code can be found **[here](https://code.visualstudio.com/docs/devcontainers/containers)**.

### **Using Docker**

If you prefer to use Docker, you can find the Dockerfile in the `.devcontainer` directory. Please refer to Docker's documentation if you need guidance on building a Docker image and running a container.

### **Using Pip or Conda**

If you wish to use Pip or Conda for managing Python dependencies, please refer to their respective documentation for instructions. It's important to ensure that all required dependencies for this project are correctly installed in your Python environment.

### Using Pip/Conda

1. Install Pytorch from official website ([link](https://pytorch.org/get-started/locally/)).

```bash
pip install \
    opencv-python \
    pytorch-lightning==1.7.7 \
    hydra-core \
    hydra-colorlog \
    hydra-optuna-sweeper \
    torchmetrics \
    pyrootutils \
    pre-commit \
    pytest \
    sh \
    omegaconf \
    rich \
    fiftyone \
    jupyter \
    wandb \
    grad-cam \
    tensorboardx \
    ipdb

pip install \
    hydra-joblib-launcher \
    gymnasium \
    mujoco \
    gym==0.25.0 \
    tianshou==0.4.11 \
    ftfy \
    regex
```

ps. We use `pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel` while other versions should also work.

Also, you may need to install MuJoCo, here we provide a simple example

- Example snippet to install MuJoCo
    
    ```bash
    curl https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz --output mujoco210.tar.gz
    mkdir ~/.mujoco
    tar -xf mujoco210.tar.gz --directory ~/.mujoco
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
    ```
    

# Run

## **Getting Started Example**

Use the following command to start a simple run:

```bash
python src/entry.py \
	experiment=sac \
	env.name=Ant-v4 \
	env.delay=4
```

## **Customization: Algorithm, Environment, Delay**

The "experiment" parameter accepts one of the following values:

- `dummy` (referred to as "vanilla SAC" in the paper)
- `oracle_critic` (referred to as "Delay-Reconciled Training" in the paper)
- `cat_mlp` (referred to as "State Augmentation - MLP" in the paper)
- `stack_rnn` (referred to as "State Augmentation - RNN" in the paper)
- `pred_detach` (referred to as "Prediction$^\dagger$" in the paper)
- `pred_nodetach` (referred to as "Prediction" in the paper)
- `encoding_detach` (referred to as "Encoding" in the paper)
- `encoding_nodetach` (referred to as "Encoding$^\dagger$" in the paper)
- `symmetric` (referred to as "Symmetric - MLP" in the paper)

The "env.name" parameter defines the environment and can be any available environment within the "gymnasium" such as:

- `Ant-v4`
- `Walker2d-v4`
- `HalfCheetah-v4`
- `and many others`

The "env.delay" parameter can be set to any non-negative integer.

Here is some examples of a customized run:

1. Using "oracle_critic" experiment, "Walker2d-v4" environment, and a delay of 5:
    
    ```
    python src/entry.py \
    	experiment=oracle_critic \
    	env.name=Walker2d-v4 \
    	env.delay=5
    
    ```
    
2. Using "cat_mlp" experiment, "HalfCheetah-v4" environment, and a delay of 3:
    
    ```
    python src/entry.py \
    	experiment=cat_mlp \
    	env.name=HalfCheetah-v4 \
    	env.delay=3
    
    ```
    
3. Using "pred_detach" experiment, "Ant-v4" environment, and a delay of 0:
    
    ```
    python src/entry.py \
    	experiment=pred_detach \
    	env.name=Ant-v4 \
    	env.delay=0
    
    ```