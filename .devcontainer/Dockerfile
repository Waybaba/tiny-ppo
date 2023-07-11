# pytorch image base
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel as base
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES},display \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# environment variables
ARG USERNAME=waybaba
ENV UDATADIR=/data \
    UPRJDIR=/code \
    UOUTDIR=/output \
    UDEVICEID=docker

# apt
RUN apt-get update && apt-get install -y \
    curl \
    git \
    sudo \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    htop \
    vim \
    libosmesa6-dev \
    libgl1-mesa-dev
    # && rm -rf /var/lib/apt/lists/*

# python packages
RUN conda install -y --name base -c conda-forge \
    tensorboard \
    pandas \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
    opencv-python \
    pytorch-lightning==1.7.7 \
    # hydra-joblib-launcher \
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
    # && rm -rf /var/lib/apt/lists/*

RUN pip install \
    hydra-joblib-launcher \
    gymnasium \
    mujoco \
    gym==0.25.0 \
    tianshou==0.4.11 \
    ftfy \
    regex



# Non-root user creation and enter
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && mkdir -p /home/$USERNAME/.vscode-server /home/$USERNAME/.vscode-server-insiders \
    && chown ${USER_UID}:${USER_GID} /home/$USERNAME/.vscode-server* \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && usermod -a -G audio,video $USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
USER $USERNAME
ENV HOME /home/$USERNAME
WORKDIR $HOME


# install mujoco
RUN mkdir ~/.mujoco
RUN wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
RUN tar -zxvf mujoco210-linux-x86_64.tar.gz
RUN mv mujoco210 ~/.mujoco/
RUN rm mujoco210-linux-x86_64.tar.gz
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

USER root

# install d4rl
RUN pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
RUN pip install dm_control mujoco-py==2.1.2.14

# install diffuser
RUN pip install einops typed-argument-parser scikit-video==1.1.11 scikit-image==0.17.2

# end
CMD	sleep infinity