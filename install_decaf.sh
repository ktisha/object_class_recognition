#!/bin/sh
git clone https://github.com/UCB-ICSI-Vision-Group/decaf-release.git
cd decaf-release
sudo apt-get install gfortran libopenblas-dev liblapack-dev

pip install six
pip install numpy
sed -i.bak s/-Wl// decaf/layers/cpp/Makefile
python setup.py build
python setup.py install