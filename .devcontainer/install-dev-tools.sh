#!/bin/bash

# sudo apt --fix-broken install -y
sudo apt update
sudo apt upgrade -y
sudo apt install software-properties-common -y
# sudo apt autoremove --purge -y
# sudo apt --fix-broken install -y

# sudo apt install python3-pip -y
# python3 -m pip install --upgrade pip
sudo pip3 install -r .devcontainer/requirements.txt

#If using Jetson Nano
#From https://qengineering.eu/install-pytorch-on-jetson-nano.html 
# Install the dependencies
# pip uninstall torch
sudo apt-get install -y python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev libgl1-mesa-glx ffmpeg libsm6 libxext6 libgl1 bc
sudo apt-get install -y build-essential libbz2-dev libssl-dev libreadline-dev \
                        libffi-dev libsqlite3-dev tk-dev
# optional scientific package headers (for Numpy, Matplotlib, SciPy, etc.)
sudo apt-get install -y libpng-dev libfreetype6-dev

sudo pip3 install future wheel mock testresources Cython gdown
# Above 58.3.0 you get version issues
sudo pip3 install setuptools==58.3.0
# Install gdown to download from Google Drive
# Download the wheel
# gdown https://drive.google.com/uc?id=1TqC6_2cwqiYacjoLhLgrZoap6-sVL2sd
# Install PyTorch 1.10.0
# pip install torch-1.10.0a0+git36449ea-cp36-cp36m-linux_aarch64.whl
# Clean up
# rm torch-1.10.0a0+git36449ea-cp36-cp36m-linux_aarch64.whl

# Installer script for pyenv
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

# Add pyenv to load path
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
[[ -d $PYENV_ROOT/bin ]] && echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Install python3.6
pyenv install 3.6.0

# Make it a vituralenv
pyenv virtualenv 3.6.0 general

# Make it globally active
pyenv global general

if [ -d "data" ]; then
  rm -r data
fi

gdown https://drive.google.com/drive/folders/1h4AyYiFY-kCU3KT-guP_QTVAcONn7VUD -O data --folder && cd data && unzip '*.zip' && rm -r .*zip

sudo apt update
# Set OPENBLAS_CORETYPE=ARMV8
# echo "export OPENBLAS_CORETYPE=ARMV8" >> ~/.bashrc
source ~/.bashrc

# Set power state of devices
# sudo nvpmodel -m 1

# Clean up
sudo pip3 cache purge
apt autoremove -y
apt clean
