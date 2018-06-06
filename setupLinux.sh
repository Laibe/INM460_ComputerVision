#!/bin/bash
sudo apt-get update;
sudo apt-get install build-essential;
sudo apt-get install cmake;
conda update conda;
conda update --all;
conda env create -f environment.yml;
source activate cv;
pip install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp35-cp35m-linux_x86_64.whl 
pip install torchvision
