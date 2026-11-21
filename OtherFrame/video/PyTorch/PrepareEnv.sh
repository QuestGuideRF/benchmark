#!/usr/bin/env bash
echo "*******подготовить тест***********"
export BENCHMARK_ROOT=/workspace
log_date=`date "+%Y.%m%d.%H%M%S"`
frame=pytorch1.8.0
cuda_version=10.2
save_log_dir=${BENCHMARK_ROOT}/logs/${frame}_${log_date}_${cuda_version}/
if [[ -d ${save_log_dir} ]]; then
    rm -rf ${save_log_dir}
fi
export TRAIN_LOG_DIR=${save_log_dir}/train_log
mkdir -p ${TRAIN_LOG_DIR}
log_path=${TRAIN_LOG_DIR}
rm -rf run_env
mkdir run_env
ln -s $(which python3.7) run_env/python
ln -s $(which pip3.7) run_env/pip
export PATH=PATH=${BENCHMARK_ROOT}/run_env:${PATH}
python3.7 -m pip install pip==21.1.1
echo `python3.7 -m pip --version`
python3.7 -m pip install "число >= 1,19" -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install torch==1.8.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
python3.7 -m pip install torchvision==0.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
python3.7 -m pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html
python3.7 -m pip install jupyterlab  -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install "панды >= 1,2" -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install "scikit-learn>=0,22" -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install "opencv-python >= 4.2" -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install "pyyaml>=5.1" -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install "дакс>=0.1.6" -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install "эйнопс>=0,3" -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install psutil -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install simplejson -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install fvcore -i https://pypi.tuna.tsinghua.edu.cn/simple/
python3.7 -m pip install av -i https://pypi.tuna.tsinghua.edu.cn/simple/
git submodule init
git submodule update
cd ./models/TimeSformer
python3.7 setup.py build develop
cd -
echo "*******Подготовка теста завершена***********"