set -ex
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
conda install pytorch torchvision -c pytorch
conda install visdom dominate -c conda-forge