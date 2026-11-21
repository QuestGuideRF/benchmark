#!/usr/bin/env bash
echo "*******подготовить тест***********"
export BENCHMARK_ROOT=/workspace
log_date=`date "+%Y.%m%d.%H%M%S"`
frame=pytorch1.0.0
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
export PATH=/workspace/run_env:${PATH}
pip install -U pip
echo `pip --version`
git clone https://github.com/lzzyzlbb/stylegan2-pytorch
cd stylegan2-pytorch
git checkout add_log
pip install torch==1.3.1
pip install torchvision==0.4.2
pip install lmdb
pip install Ninja
pip install tqdm
mkdir -p data
wget https://paddlegan.bj.bcebos.com/datasets/stylegan_process.tar \
    -O data/ffhq.tar
tar -vxf data/ffhq.tar -C data/
echo "*******Подготовка теста завершена***********"