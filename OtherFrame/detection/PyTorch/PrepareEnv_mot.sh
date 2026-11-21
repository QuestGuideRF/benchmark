#!/usr/bin/env bash
echo "*******подготовить тест***********"
export BENCHMARK_ROOT=/workspace
log_date=`date "+%Y.%m%d.%H%M%S"`
frame=pytorch1.10.1
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
python3.7 -m pip install -U pip
echo `python3.7 -m pip --version`
python3.7 -m pip install ${ALL_PATH}/other_frame_whls/*
pip install -r requirement.txt
git submodule init
git submodule update
wget -nc -P ${BENCHMARK_ROOT}/data/mot/ https://paddledet.bj.bcebos.com/data/mot_benchmark.tar
cd ${BENCHMARK_ROOT}/models/jde
mkdir -p dataset/mot
cp ${BENCHMARK_ROOT}/data/mot/mot_benchmark.tar dataset/mot
cd ./dataset/mot/ && tar -xvf mot_benchmark.tar && mv -u mot_benchmark/* .
rm -rf mot_benchmark/
cd ${BENCHMARK_ROOT}/models/fairmot/src
mkdir -p dataset/mot
cp ${BENCHMARK_ROOT}/data/mot/mot_benchmark.tar dataset/mot
cd ./dataset/mot/ && tar -xvf mot_benchmark.tar && mv -u mot_benchmark/* .
rm -rf mot_benchmark/
cd ${BENCHMARK_ROOT}
cd ${BENCHMARK_ROOT}/models/jde
python3.7 -m pip install -r requirements.txt
cd ${BENCHMARK_ROOT}/models/fairmot
python3.7 -m pip install -r requirements.txt
cd ${BENCHMARK_ROOT}/models/jde
mkdir -p weights
cd weights
wget https://paddledet.bj.bcebos.com/models/pretrained/darknet53.conv.74
cd ${BENCHMARK_ROOT}/models/fairmot
mkdir -p models
cd models
wget https://paddledet.bj.bcebos.com/models/pretrained/ctdet_coco_dla_2x.pth
cd ${BENCHMARK_ROOT}
echo "*******Подготовка теста завершена***********"