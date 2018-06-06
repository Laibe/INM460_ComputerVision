#!/bin/bash
# Requirements CLT for Xcode
# Anaconda 3+ 
# install homebrew package manager for macOS
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)";
# install cmake via homebrew - cmake is required to build the dlib library
brew install cmake;
# install tesseract for OCR
brew install tesseract;
# update anaconda
conda update conda;
conda update --all;
# create a new environment
conda env create -f environment.yml;
# activate the environment
source activate cv;
pip install http://download.pytorch.org/whl/torch-0.3.1-cp35-cp35m-macosx_10_6_x86_64.whl; 
pip install torchvision;
